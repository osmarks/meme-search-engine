package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"regexp"
	"strings"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/samber/lo"
	"github.com/titanous/json5"
)

const CALLBACK_REGEX string = ">AF_initDataCallback\\(({key: 'ds:1'.*?)\\);</script>"

type SegmentCoords struct {
	x int
	y int
	w int
	h int
}

type Segment struct {
	coords SegmentCoords
	text   string
}

type ScanResult []Segment

// TODO coordinates are negative sometimes and I think they shouldn't be
func rationalizeCoordsFormat1(imageW float64, imageH float64, centerXFraction float64, centerYFraction float64, widthFraction float64, heightFraction float64) SegmentCoords {
	return SegmentCoords{
		x: int(math.Round((centerXFraction - widthFraction/2) * imageW)),
		y: int(math.Round((centerYFraction - heightFraction/2) * imageH)),
		w: int(math.Round(widthFraction * imageW)),
		h: int(math.Round(heightFraction * imageH)),
	}
}

func scanImageChunk(image []byte, imageWidth int, imageHeight int) (ScanResult, error) {
	var result ScanResult
	timestamp := time.Now().UnixMicro()
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	defer w.Close()
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="encoded_image"; filename="ocr%d.png"`, timestamp))
	h.Set("Content-Type", "image/png")
	fw, err := w.CreatePart(h)
	if err != nil {
		return result, err
	}
	fw.Write(image)
	w.Close()

	req, err := http.NewRequest("POST", fmt.Sprintf("https://lens.google.com/v3/upload?stcs=%d", timestamp), &b)
	if err != nil {
		return result, err
	}
	req.Header.Add("User-Agent", "Mozilla/5.0 (Linux; Android 13; RMX3771) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.144 Mobile Safari/537.36")
	req.AddCookie(&http.Cookie{
		Name:  "SOCS",
		Value: "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg",
	})
	req.Header.Set("Content-Type", w.FormDataContentType())
	client := http.Client{}
	res, err := client.Do(req)
	if err != nil {
		return result, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return result, err
	}
	re, _ := regexp.Compile(CALLBACK_REGEX)
	matches := re.FindStringSubmatch(string(body[:]))
	if len(matches) == 0 {
		return result, fmt.Errorf("invalid API response")
	}
	match := matches[1]
	var lensObject map[string]interface{}
	err = json5.Unmarshal([]byte(match), &lensObject)
	if err != nil {
		return result, err
	}

	if _, ok := lensObject["errorHasStatus"]; ok {
		return result, errors.New("lens failed")
	}

	root := lensObject["data"].([]interface{})

	var textSegments []string
	var textRegions []SegmentCoords

	// I don't know why Google did this.
	// Text segments are in one place and their locations are in another, using a very strange coordinate system.
	// At least I don't need whatever is contained in the base64 parts (which I assume are protobufs).
	// TODO: on a few images, this seems to not work for some reason.
	defer func() {
		if r := recover(); r != nil {
			// https://github.com/dimdenGD/chrome-lens-ocr/blob/main/src/core.js#L316 has code for a fallback text segment read mode.
			// In testing, this proved unnecessary (quirks of the HTTP request? I don't know), and this only happens on textless images.
			textSegments = []string{}
			textRegions = []SegmentCoords{}
		}
	}()

	textSegmentsRaw := root[3].([]interface{})[4].([]interface{})[0].([]interface{})[0].([]interface{})
	textRegionsRaw := root[2].([]interface{})[3].([]interface{})[0].([]interface{})
	for _, x := range textRegionsRaw {
		if strings.HasPrefix(x.([]interface{})[11].(string), "text:") {
			rawCoords := x.([]interface{})[1].([]interface{})
			coords := rationalizeCoordsFormat1(float64(imageWidth), float64(imageHeight), rawCoords[0].(float64), rawCoords[1].(float64), rawCoords[2].(float64), rawCoords[3].(float64))
			textRegions = append(textRegions, coords)
		}
	}
	for _, x := range textSegmentsRaw {
		textSegment := x.(string)
		textSegments = append(textSegments, textSegment)
	}

	return lo.Map(lo.Zip2(textSegments, textRegions), func(x lo.Tuple2[string, SegmentCoords], _ int) Segment {
		return Segment{
			text:   x.A,
			coords: x.B,
		}
	}), nil
}

const MAX_DIM int = 1024

func scanImage(image *vips.ImageRef) (ScanResult, error) {
	result := ScanResult{}
	width := image.Width()
	height := image.Height()
	if width > MAX_DIM {
		width = MAX_DIM
		height = int(math.Round(float64(height) * (float64(width) / float64(image.Width()))))
	}
	downscaled, err := image.Copy()
	if err != nil {
		return result, err
	}
	downscaled.Resize(float64(width)/float64(image.Width()), vips.KernelLanczos3)
	for y := 0; y < height; y += MAX_DIM {
		chunkHeight := MAX_DIM
		if y+chunkHeight > height {
			chunkHeight = height - y
		}
		chunk, err := image.Copy() // TODO this really really should not be in-place
		if err != nil {
			return result, err
		}
		err = chunk.ExtractArea(0, y, width, height)
		if err != nil {
			return result, err
		}
		buf, _, err := chunk.ExportPng(&vips.PngExportParams{})
		if err != nil {
			return result, err
		}
		res, err := scanImageChunk(buf, width, chunkHeight)
		if err != nil {
			return result, err
		}
		for _, segment := range res {
			result = append(result, Segment{
				text: segment.text,
				coords: SegmentCoords{
					y: segment.coords.y + y,
					x: segment.coords.x,
					w: segment.coords.w,
					h: segment.coords.h,
				},
			})
		}
	}

	return result, nil
}

/*
async def scan_image_chunk(sess, image):
    # send data to inscrutable undocumented Google service
    # https://github.com/AuroraWright/owocr/blob/master/owocr/ocr.py#L193
    async with aiohttp.ClientSession() as sess:
        data = aiohttp.FormData()
        data.add_field(
            "encoded_image",
            encode_img(image),
            filename="ocr" + str(timestamp) + ".png",
            content_type="image/png"
        )
        async with sess.post(url, headers=headers, cookies=cookies, data=data, timeout=10) as res:
            body = await res.text()

    # I really worry about Google sometimes. This is not a sensible format.
    match = CALLBACK_REGEX.search(body)
    if match == None:
        raise ValueError("Invalid callback")

    lens_object = pyjson5.loads(match.group(1))
    if "errorHasStatus" in lens_object:
        raise RuntimeError("Lens failed")

    text_segments = []
    text_regions = []

    root = lens_object["data"]

    # I don't know why Google did this.
    # Text segments are in one place and their locations are in another, using a very strange coordinate system.
    # At least I don't need whatever is contained in the base64 partss (which I assume are protobufs).
    # TODO: on a few images, this seems to not work for some reason.
    try:
        text_segments = root[3][4][0][0]
        text_regions = [ rationalize_coords_format1(image.width, image.height, *x[1]) for x in root[2][3][0] if x[11].startswith("text:") ]
    except (KeyError, IndexError):
        # https://github.com/dimdenGD/chrome-lens-ocr/blob/main/src/core.js#L316 has code for a fallback text segment read mode.
        # In testing, this proved unnecessary (quirks of the HTTP request? I don't know), and this only happens on textless images.
        return [], []

    return text_segments, text_regions

MAX_SCAN_DIM = 1000 # not actually true but close enough
def chunk_image(image: Image):
    chunks = []
    # Cut image down in X axis (I'm assuming images aren't too wide to scan in downscaled form because merging text horizontally would be annoying)
    if image.width > MAX_SCAN_DIM:
        image = image.resize((MAX_SCAN_DIM, round(image.height * (image.width / MAX_SCAN_DIM))), Image.LANCZOS)
    for y in range(0, image.height, MAX_SCAN_DIM):
        chunks.append(image.crop((0, y, image.width, min(y + MAX_SCAN_DIM, image.height))))
    return chunks

async def scan_chunks(sess: aiohttp.ClientSession, chunks: [Image]):
    # If text happens to be split across the cut line it won't get read.
    # This is because doing overlap read areas would be really annoying.
    text = ""
    regions = []
    for chunk in chunks:
        new_segments, new_regions = await scan_image_chunk(sess, chunk)
        for segment in new_segments:
            text += segment + "\n"
        for i, (segment, region) in enumerate(zip(new_segments, new_regions)):
            regions.append({ **region, "y": region["y"] + (MAX_SCAN_DIM * i), "text": segment })
    return text, regions

async def scan_image(sess: aiohttp.ClientSession, image: Image):
    return await scan_chunks(sess, chunk_image(image))

if __name__ == "__main__":
    async def main():
        async with aiohttp.ClientSession() as sess:
            print(await scan_image(sess, Image.open("/data/public/memes-or-something/linear-algebra-chess.png")))
    asyncio.run(main())
*/
