package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"time"

	"github.com/DataIntelligenceCrew/go-faiss"
	"github.com/h2non/bimg"
	"github.com/jmoiron/sqlx"
	_ "github.com/mattn/go-sqlite3"
	"github.com/samber/lo"
	"github.com/vmihailenco/msgpack"
	"github.com/x448/float16"
	"golang.org/x/sync/errgroup"
)

type Config struct {
	ClipServer       string `json:"clip_server"`
	DbPath           string `json:"db_path"`
	Port             int16  `json:"port"`
	Files            string `json:"files"`
	EnableOCR        bool   `json:"enable_ocr"`
	ThumbsPath       string `json:"thumbs_path"`
	EnableThumbnails bool   `json:"enable_thumbs"`
}

type Index struct {
	vectors     *faiss.IndexImpl
	filenames   []string
	formatCodes []int64
	formatNames []string
}

var schema = `
CREATE TABLE IF NOT EXISTS files (
	filename TEXT PRIMARY KEY,
	embedding_time INTEGER,
	ocr_time INTEGER,
	thumbnail_time INTEGER,
	embedding BLOB,
	ocr TEXT,
	raw_ocr_segments BLOB,
	thumbnails BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5 (
	filename,
	ocr,
	tokenize='unicode61 remove_diacritics 2',
	content='ocr'
);

CREATE TRIGGER IF NOT EXISTS ocr_fts_ins AFTER INSERT ON files BEGIN
	INSERT INTO ocr_fts (rowid, filename, ocr) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER DELETE ON files BEGIN
	INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER UPDATE ON files BEGIN
	INSERT INTO ocr_fts (ocr_fts, rowid, filename, ocr) VALUES ('delete', old.rowid, old.filename, COALESCE(old.ocr, ''));
	INSERT INTO ocr_fts (rowid, filename, text) VALUES (new.rowid, new.filename, COALESCE(new.ocr, ''));
END;
`

type FileRecord struct {
	Filename       string `db:"filename"`
	EmbedTime      int64  `db:"embedding_time"`
	OcrTime        int64  `db:"ocr_time"`
	ThumbnailTime  int64  `db:"thumbnail_time"`
	Embedding      []byte `db:"embedding"`
	Ocr            string `db:"ocr"`
	RawOcrSegments []byte `db:"raw_ocr_segments"`
	Thumbnails     []byte `db:"thumbnails"`
}

type InferenceServerConfig struct {
	BatchSize     uint   `msgpack:"batch"`
	ImageSize     []uint `msgpack:"image_size"`
	EmbeddingSize uint   `msgpack:"embedding_size"`
}

func decodeMsgpackFrom[O interface{}](resp *http.Response) (O, error) {
	var result O
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return result, err
	}
	err = msgpack.Unmarshal(respData, &result)
	return result, err
}

func queryClipServer[I interface{}, O interface{}](config Config, path string, data I) (O, error) {
	var result O
	b, err := msgpack.Marshal(data)
	if err != nil {
		return result, err
	}
	resp, err := http.Post(config.ClipServer+path, "application/msgpack", bytes.NewReader(b))
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()
	return decodeMsgpackFrom[O](resp)
}

type LoadedImage struct {
	image        *bimg.Image
	filename     string
	originalSize int
}

type EmbeddingInput struct {
	image    []byte
	filename string
}

type EmbeddingRequest struct {
	Images [][]byte `msgpack:"images"`
	Text   []string `msgpack:"text"`
}

type EmbeddingResponse = [][]byte

func timestamp() int64 {
	return time.Now().UnixMicro()
}

type ImageFormatConfig struct {
	targetWidth    int
	targetFilesize int
	quality        int
	format         bimg.ImageType
	extension      string
}

func generateFilenameHash(filename string) string {
	hasher := fnv.New128()
	hasher.Write([]byte(filename))
	hash := hasher.Sum(make([]byte, 0))
	return base64.RawURLEncoding.EncodeToString(hash)
}

func generateThumbnailFilename(filename string, formatName string, formatConfig ImageFormatConfig) string {
	return fmt.Sprintf("%s%s.%s", generateFilenameHash(filename), formatName, formatConfig.extension)
}

func initializeDatabase(config Config) (*sqlx.DB, error) {
	db, err := sqlx.Connect("sqlite3", config.DbPath)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec("PRAGMA busy_timeout = 2000; PRAGMA journal_mode = WAL")
	if err != nil {
		return nil, err
	}
	return db, nil
}

func imageFormats(config Config) map[string]ImageFormatConfig {
	return map[string]ImageFormatConfig{
		"jpegl": {
			targetWidth: 800,
			quality:     70,
			format:      bimg.JPEG,
			extension:   "jpg",
		},
		"jpegh": {
			targetWidth: 1600,
			quality:     80,
			format:      bimg.JPEG,
			extension:   "jpg",
		},
		"jpeg256kb": {
			targetWidth:    500,
			targetFilesize: 256000,
			format:         bimg.JPEG,
			extension:      "jpg",
		},
		"avifh": {
			targetWidth: 1600,
			quality:     80,
			format:      bimg.AVIF,
			extension:   "avif",
		},
		"avifl": {
			targetWidth: 800,
			quality:     30,
			format:      bimg.AVIF,
			extension:   "avif",
		},
	}
}

func ingestFiles(config Config, backend InferenceServerConfig) error {
	var wg errgroup.Group
	var iwg errgroup.Group

	// We assume everything is either a modern browser (low-DPI or high-DPI), an ancient browser or a ComputerCraft machine abusing Extra Utilities 2 screens.
	var formats = imageFormats(config)

	db, err := initializeDatabase(config)
	if err != nil {
		return err
	}
	defer db.Close()

	toProcess := make(chan FileRecord, 100)
	toEmbed := make(chan EmbeddingInput, backend.BatchSize)
	toThumbnail := make(chan LoadedImage, 30)
	toOCR := make(chan LoadedImage, 30)
	embedBatches := make(chan []EmbeddingInput, 1)

	// image loading and preliminary resizing
	for range runtime.NumCPU() {
		iwg.Go(func() error {
			for record := range toProcess {
				path := filepath.Join(config.Files, record.Filename)
				buffer, err := bimg.Read(path)
				if err != nil {
					log.Println("could not read ", record.Filename)
				}
				img := bimg.NewImage(buffer)
				if record.Embedding == nil {
					resized, err := img.Process(bimg.Options{
						Width:          int(backend.ImageSize[0]),
						Height:         int(backend.ImageSize[1]),
						Force:          true,
						Type:           bimg.PNG,
						Interpretation: bimg.InterpretationSRGB,
					})
					if err != nil {
						log.Println("resize failure", record.Filename, err)
					} else {
						toEmbed <- EmbeddingInput{
							image:    resized,
							filename: record.Filename,
						}
					}
				}
				if record.Thumbnails == nil && config.EnableThumbnails {
					toThumbnail <- LoadedImage{
						image:        img,
						filename:     record.Filename,
						originalSize: len(buffer),
					}
				}
				if record.RawOcrSegments == nil && config.EnableOCR {
					toOCR <- LoadedImage{
						image:    img,
						filename: record.Filename,
					}
				}
			}
			return nil
		})
	}

	if config.EnableThumbnails {
		for range runtime.NumCPU() {
			wg.Go(func() error {
				for image := range toThumbnail {
					generatedFormats := make([]string, 0)
					for formatName, formatConfig := range formats {
						var err error
						var resized []byte
						if formatConfig.targetFilesize != 0 {
							lb := 1
							ub := 100
							for {
								quality := (lb + ub) / 2
								resized, err = image.image.Process(bimg.Options{
									Width:         formatConfig.targetWidth,
									Type:          formatConfig.format,
									Speed:         4,
									Quality:       quality,
									StripMetadata: true,
									Enlarge:       false,
								})
								if len(resized) > image.originalSize {
									ub = quality
								} else {
									lb = quality + 1
								}
								if lb >= ub {
									break
								}
							}
						} else {
							resized, err = image.image.Process(bimg.Options{
								Width:         formatConfig.targetWidth,
								Type:          formatConfig.format,
								Speed:         4,
								Quality:       formatConfig.quality,
								StripMetadata: true,
								Enlarge:       false,
							})
						}
						if err != nil {
							log.Println("thumbnailing failure", image.filename, err)
							continue
						}
						if len(resized) < image.originalSize {
							generatedFormats = append(generatedFormats, formatName)
							err = bimg.Write(filepath.Join(config.ThumbsPath, generateThumbnailFilename(image.filename, formatName, formatConfig)), resized)
							if err != nil {
								return err
							}
						}
					}
					formatsData, err := msgpack.Marshal(generatedFormats)
					if err != nil {
						return err
					}
					_, err = db.Exec("UPDATE files SET thumbnails = ?, thumbnail_time = ? WHERE filename = ?", formatsData, timestamp(), image.filename)
					if err != nil {
						return err
					}
				}
				return nil
			})
		}
	}

	if config.EnableOCR {
		for range 100 {
			wg.Go(func() error {
				for image := range toOCR {
					scan, err := scanImage(image.image)
					if err != nil {
						log.Println("OCR failure", image.filename, err)
						continue
					}
					ocrText := ""
					for _, segment := range scan {
						ocrText += segment.text
						ocrText += "\n"
					}
					ocrData, err := msgpack.Marshal(scan)
					if err != nil {
						return err
					}
					_, err = db.Exec("UPDATE files SET ocr = ?, raw_ocr_segments = ?, ocr_time = ? WHERE filename = ?", ocrText, ocrData, timestamp(), image.filename)
					if err != nil {
						return err
					}
				}
				return nil
			})
		}
	}

	wg.Go(func() error {
		buffer := make([]EmbeddingInput, 0, backend.BatchSize)
		for input := range toEmbed {
			buffer = append(buffer, input)
			if len(buffer) == int(backend.BatchSize) {
				embedBatches <- buffer
				buffer = make([]EmbeddingInput, 0, backend.BatchSize)
			}
		}
		if len(buffer) > 0 {
			embedBatches <- buffer
		}
		close(embedBatches)
		return nil
	})

	for range 3 {
		wg.Go(func() error {
			for batch := range embedBatches {
				result, err := queryClipServer[EmbeddingRequest, EmbeddingResponse](config, "", EmbeddingRequest{
					Images: lo.Map(batch, func(item EmbeddingInput, _ int) []byte { return item.image }),
				})
				if err != nil {
					return err
				}

				tx, err := db.Begin()
				if err != nil {
					return err
				}
				for i, vector := range result {
					_, err = tx.Exec("UPDATE files SET embedding_time = ?, embedding = ? WHERE filename = ?", timestamp(), vector, batch[i].filename)
					if err != nil {
						return err
					}
				}
				err = tx.Commit()
				if err != nil {
					return err
				}
			}
			return nil
		})
	}

	filenamesOnDisk := make(map[string]struct{})

	err = filepath.WalkDir(config.Files, func(path string, d os.DirEntry, err error) error {
		filename := strings.TrimPrefix(path, config.Files)
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		filenamesOnDisk[filename] = struct{}{}
		records := []FileRecord{}
		err = db.Select(&records, "SELECT * FROM files WHERE filename = ?", filename)
		if err != nil {
			return err
		}
		stat, err := d.Info()
		if err != nil {
			return err
		}
		modtime := stat.ModTime().UnixMicro()
		if len(records) == 0 || modtime > records[0].EmbedTime || modtime > records[0].OcrTime || modtime > records[0].ThumbnailTime {
			_, err = db.Exec("INSERT OR IGNORE INTO files VALUES (?, 0, 0, 0, '', '', '', '')", filename)
			if err != nil {
				return err
			}
			record := FileRecord{
				Filename: filename,
			}
			if len(records) > 0 {
				record = records[0]
			}
			if modtime > record.EmbedTime || len(record.Embedding) == 0 {
				record.Embedding = nil
			}
			if modtime > record.OcrTime || len(record.RawOcrSegments) == 0 {
				record.RawOcrSegments = nil
			}
			if modtime > record.ThumbnailTime || len(record.Thumbnails) == 0 {
				record.Thumbnails = nil
			}
			toProcess <- record
		}
		return nil
	})
	if err != nil {
		return err
	}
	close(toProcess)

	err = iwg.Wait()
	close(toEmbed)
	close(toThumbnail)
	if err != nil {
		return err
	}
	err = wg.Wait()
	if err != nil {
		return err
	}

	rows, err := db.Queryx("SELECT filename FROM files")
	if err != nil {
		return err
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	for rows.Next() {
		var filename string
		err := rows.Scan(&filename)
		if err != nil {
			return err
		}
		if _, ok := filenamesOnDisk[filename]; !ok {
			_, err = tx.Exec("DELETE FROM files WHERE filename = ?", filename)
			if err != nil {
				return err
			}
		}
	}
	if err = tx.Commit(); err != nil {
		return err
	}

	return nil
}

const INDEX_ADD_BATCH = 512

func buildIndex(config Config, backend InferenceServerConfig) (Index, error) {
	var index Index

	db, err := initializeDatabase(config)
	if err != nil {
		return index, err
	}
	defer db.Close()

	newFAISSIndex, err := faiss.IndexFactory(int(backend.EmbeddingSize), "SQfp16", faiss.MetricInnerProduct)
	if err != nil {
		return index, err
	}
	index.vectors = newFAISSIndex

	var count int
	err = db.Get(&count, "SELECT COUNT(*) FROM files")
	if err != nil {
		return index, err
	}

	index.filenames = make([]string, 0, count)
	index.formatCodes = make([]int64, 0, count)
	buffer := make([]float32, 0, INDEX_ADD_BATCH*backend.EmbeddingSize)
	index.formatNames = make([]string, 0, 5)

	record := FileRecord{}
	rows, err := db.Queryx("SELECT * FROM files")
	if err != nil {
		return index, err
	}
	for rows.Next() {
		err := rows.StructScan(&record)
		if err != nil {
			return index, err
		}
		if len(record.Embedding) > 0 {
			index.filenames = append(index.filenames, record.Filename)
			for i := 0; i < len(record.Embedding); i += 2 {
				buffer = append(buffer, float16.Frombits(uint16(record.Embedding[i])+uint16(record.Embedding[i+1])<<8).Float32())
			}
			if len(buffer) == cap(buffer) {
				index.vectors.Add(buffer)
				buffer = make([]float32, 0, INDEX_ADD_BATCH*backend.EmbeddingSize)
			}

			formats := make([]string, 0, 5)
			if len(record.Thumbnails) > 0 {
				err := msgpack.Unmarshal(record.Thumbnails, &formats)
				if err != nil {
					return index, err
				}
			}

			formatCode := int64(0)
			for _, formatString := range formats {
				found := false
				for i, name := range index.formatNames {
					if name == formatString {
						formatCode |= 1 << i
						found = true
						break
					}
				}
				if !found {
					newIndex := len(index.formatNames)
					formatCode |= 1 << newIndex
					index.formatNames = append(index.formatNames, formatString)
				}
			}
			index.formatCodes = append(index.formatCodes, formatCode)
		}
	}
	if len(buffer) > 0 {
		index.vectors.Add(buffer)
	}

	return index, nil
}

func decodeFP16Buffer(buf []byte) []float32 {
	out := make([]float32, 0, len(buf)/2)
	for i := 0; i < len(buf); i += 2 {
		out = append(out, float16.Frombits(uint16(buf[i])+uint16(buf[i+1])<<8).Float32())
	}
	return out
}

type EmbeddingVector []float32

type QueryResult struct {
	Matches    [][]interface{}   `json:"matches"`
	Formats    []string          `json:"formats"`
	Extensions map[string]string `json:"extensions"`
}

// this terrible language cannot express tagged unions
type QueryTerm struct {
	Embedding *EmbeddingVector `json:"embedding"`
	Image     *string          `json:"image"` // base64
	Text      *string          `json:"text"`
	Weight    *float32         `json:"weight"`
}

type QueryRequest struct {
	Terms []QueryTerm `json:"terms"`
	K     *int        `json:"k"`
}

func queryIndex(index *Index, query EmbeddingVector, k int) (QueryResult, error) {
	var qr QueryResult
	distances, ids, err := index.vectors.Search(query, int64(k))
	if err != nil {
		return qr, err
	}
	items := lo.Map(lo.Zip2(distances, ids), func(x lo.Tuple2[float32, int64], i int) []interface{} {
		return []interface{}{
			x.A,
			index.filenames[x.B],
			generateFilenameHash(index.filenames[x.B]),
			index.formatCodes[x.B],
		}
	})

	return QueryResult{
		Matches: items,
		Formats: index.formatNames,
	}, nil
}

func handleRequest(config Config, backendConfig InferenceServerConfig, index *Index, w http.ResponseWriter, req *http.Request) error {
	if req.Body == nil {
		io.WriteString(w, "OK") // health check
		return nil
	}
	dec := json.NewDecoder(req.Body)
	var qreq QueryRequest
	err := dec.Decode(&qreq)
	if err != nil {
		return err
	}

	totalEmbedding := make(EmbeddingVector, backendConfig.EmbeddingSize)

	imageBatch := make([][]byte, 0)
	imageWeights := make([]float32, 0)
	textBatch := make([]string, 0)
	textWeights := make([]float32, 0)

	for _, term := range qreq.Terms {
		if term.Image != nil {
			bytes, err := base64.StdEncoding.DecodeString(*term.Image)
			if err != nil {
				return err
			}
			loaded := bimg.NewImage(bytes)
			resized, err := loaded.Process(bimg.Options{
				Width:          int(backendConfig.ImageSize[0]),
				Height:         int(backendConfig.ImageSize[1]),
				Force:          true,
				Type:           bimg.PNG,
				Interpretation: bimg.InterpretationSRGB,
			})
			if err != nil {
				return err
			}
			imageBatch = append(imageBatch, resized)
			if term.Weight != nil {
				imageWeights = append(imageWeights, *term.Weight)
			} else {
				imageWeights = append(imageWeights, 1)
			}
		}
		if term.Text != nil {
			textBatch = append(textBatch, *term.Text)
			if term.Weight != nil {
				textWeights = append(textWeights, *term.Weight)
			} else {
				textWeights = append(textWeights, 1)
			}
		}
		if term.Embedding != nil {
			weight := float32(1.0)
			if term.Weight != nil {
				weight = *term.Weight
			}
			for i := 0; i < int(backendConfig.EmbeddingSize); i += 1 {
				totalEmbedding[i] += (*term.Embedding)[i] * weight
			}
		}
	}

	if len(imageBatch) > 0 {
		embs, err := queryClipServer[EmbeddingRequest, EmbeddingResponse](config, "/", EmbeddingRequest{
			Images: imageBatch,
		})
		if err != nil {
			return err
		}
		for j, emb := range embs {
			embd := decodeFP16Buffer(emb)
			for i := 0; i < int(backendConfig.EmbeddingSize); i += 1 {
				totalEmbedding[i] += embd[i] * imageWeights[j]
			}
		}
	}
	if len(textBatch) > 0 {
		embs, err := queryClipServer[EmbeddingRequest, EmbeddingResponse](config, "/", EmbeddingRequest{
			Text: textBatch,
		})
		if err != nil {
			return err
		}
		for j, emb := range embs {
			embd := decodeFP16Buffer(emb)
			for i := 0; i < int(backendConfig.EmbeddingSize); i += 1 {
				totalEmbedding[i] += embd[i] * textWeights[j]
			}
		}
	}

	k := 1000
	if qreq.K != nil {
		k = *qreq.K
	}

	w.Header().Add("Content-Type", "application/json")
	enc := json.NewEncoder(w)

	qres, err := queryIndex(index, totalEmbedding, k)

	qres.Extensions = make(map[string]string)
	for k, v := range imageFormats(config) {
		qres.Extensions[k] = v.extension
	}

	if err != nil {
		return err
	}

	err = enc.Encode(qres)
	if err != nil {
		return err
	}
	return nil
}

func init() {
	os.Setenv("VIPS_WARNING", "FALSE") // this does not actually work
	bimg.VipsCacheSetMax(0)
	bimg.VipsCacheSetMaxMem(0)
}

func main() {
	content, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal("config file unreadable ", err)
	}
	var config Config
	err = json.Unmarshal(content, &config)
	if err != nil {
		log.Fatal("config file wrong ", err)
	}
	fmt.Println(config)

	db, err := sqlx.Connect("sqlite3", config.DbPath)
	if err != nil {
		log.Fatal("DB connection failure ", db)
	}
	db.MustExec(schema)

	var backend InferenceServerConfig
	for {
		resp, err := http.Get(config.ClipServer + "/config")
		if err != nil {
			log.Println("backend failed (fetch) ", err)
		}
		backend, err = decodeMsgpackFrom[InferenceServerConfig](resp)
		resp.Body.Close()
		if err != nil {
			log.Println("backend failed (parse) ", err)
		} else {
			break
		}
		time.Sleep(time.Second)
	}

	requestIngest := make(chan struct{}, 1)

	var index *Index
	// maybe this ought to be mutexed?
	var lastError *error
	// there's not a neat way to reusably broadcast to multiple channels, but I *can* abuse WaitGroups probably
	// this might cause horrible concurrency issues, but you brought me to this point, Go designers
	var wg sync.WaitGroup

	go func() {
		for {
			wg.Add(1)
			log.Println("ingest running")
			err := ingestFiles(config, backend)
			if err != nil {
				log.Println("ingest failed ", err)
				lastError = &err
			} else {
				newIndex, err := buildIndex(config, backend)
				if err != nil {
					log.Println("index build failed ", err)
					lastError = &err
				} else {
					lastError = nil
					index = &newIndex
				}
			}
			wg.Done()
			<-requestIngest
		}
	}()
	newIndex, err := buildIndex(config, backend)
	index = &newIndex
	if err != nil {
		log.Fatal("index build failed ", err)
	}

	http.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		w.Header().Add("Access-Control-Allow-Origin", "*")
		w.Header().Add("Access-Control-Allow-Headers", "Content-Type")
		if req.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		err := handleRequest(config, backend, index, w, req)
		if err != nil {
			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(500)
			json.NewEncoder(w).Encode(map[string]string{
				"error": err.Error(),
			})
		}
	})
	http.HandleFunc("/reload", func(w http.ResponseWriter, req *http.Request) {
		if req.Method == "POST" {
			log.Println("requesting index reload")
			select {
			case requestIngest <- struct{}{}:
			default:
			}
			wg.Wait()
			if lastError == nil {
				w.Write([]byte("OK"))
			} else {
				w.WriteHeader(500)
				w.Write([]byte((*lastError).Error()))
			}
		}
	})
	http.HandleFunc("/profile", func(w http.ResponseWriter, req *http.Request) {
		f, err := os.Create("mem.pprof")
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		defer f.Close()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		log.Printf("Memory usage: Alloc=%v, TotalAlloc=%v, Sys=%v", m.Alloc, m.TotalAlloc, m.Sys)
		log.Println(bimg.VipsMemory())
		bimg.VipsDebugInfo()
		runtime.GC() // Trigger garbage collection
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
	})
	log.Println("starting server")
	http.ListenAndServe(fmt.Sprintf(":%d", config.Port), nil)
}
