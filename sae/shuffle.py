# claude output

import pyarrow as pa
import numpy as np
import pyarrow.compute as pc

def shuffle_arrow_file(input_file, output_file, chunk_size=100000):
    # Open the input file
    with pa.memory_map(input_file, 'r') as source:
        reader = pa.ipc.open_file(source)
        
        # Get the schema and create a writer for the output file
        schema = reader.schema
        with pa.OSFile(output_file, 'wb') as sink:
            writer = pa.ipc.new_file(sink, schema)
            
            # Calculate total number of rows
            total_rows = reader.num_record_batches
            all_batches = [reader.get_batch(i) for i in range(total_rows)]
            total_rows = sum(batch.num_rows for batch in all_batches)
            
            # Generate shuffled indices
            indices = np.random.permutation(total_rows)
            
            # Process in chunks
            for i in range(0, total_rows, chunk_size):
                # Get indices for this chunk
                chunk_indices = indices[i:i+chunk_size]
                
                # Take rows using these indices
                chunk_data = []
                for idx in chunk_indices:
                    batch_idx = 0
                    row_idx = idx
                    while row_idx >= all_batches[batch_idx].num_rows:
                        row_idx -= all_batches[batch_idx].num_rows
                        batch_idx += 1
                    chunk_data.append(all_batches[batch_idx].slice(row_idx, 1))
                
                chunk = pa.Table.from_batches(chunk_data)
                
                # Write the chunk
                writer.write_table(chunk)
            
            # Close the writer
            writer.close()

# Usage
input_file = "../../sample_1m.arrow"
output_file = "shuffled_trainset.arrow"
shuffle_arrow_file(input_file, output_file)
