module MdiscrSegmented
    (
     segmMdiscr
    ) where

import Helpers
import Data.List(zipWith4)

-- Segmented version.
segmMdiscr :: Int -> (a -> Int) -> [Int] -> [a] -> ([Int], [a])
segmMdiscr m discr segment_sizes arr =
    let
        -- The length of the input array.
        n = length arr
        
        -- Sizes extended inside each segment.
        sizes_extended = segmScanInc (+) 0 segment_sizes segment_sizes
        
        -- Sizes accumulated across segment boundaries (in order to produce
        -- the offsets to the segment starts).
        sizes_accumulated = scanInc (+) 0 segment_sizes
        
        -- Offsets to the segment starts.
        segment_offsets = zipWith (\size_accum size_ext -> size_accum - size_ext)
                          sizes_accumulated
                          sizes_extended
        
        -- Find the equivalence classes using the discriminator.
        classes = map discr arr
        
        -- Here, we could assert that all the classes are between 0 and m-1.
        
        -- Turn the classes into columns (lists).
        columns = map (createColumn m) classes
        
        -- Scan the columns.
        scan_results = segmScanInc (zipWith (+)) (replicate m 0) segment_sizes columns
        
        -- Distribute the last elements of the scan_results (i.e., the reductions)
        -- backwards across the segments.
        reductions = zipWith (\size_ext offset -> scan_results !! (size_ext+offset-1))
                     sizes_extended segment_offsets
        
        -- The reductions count the number of occurrences of each class. With
        -- the exclusive scan, we get the offsets that we must add to each class
        -- inside the segments.
        class_offsets = map (scanExc (+) 0) reductions
        
        -- Get the indices by selecting the appropriate entries from the columns
        -- and adding the appropriate offsets.
        indices =
            zipWith4 (\k scan_result class_offset segment_offset ->
                      let
                          -- Select the k'th entries.
                          scan_result_k = scan_result !! k
                          class_offset_k = class_offset !! k
                      in
                        -- Add offsets. Subtract 1 to make it 0-indexed.
                        scan_result_k + class_offset_k + segment_offset - 1
                     )
            classes scan_results class_offsets segment_offsets
        
        -- Compute the resulting array based on the indices.
        new_arr = permute indices arr
        
        -- A "iota segment_size" for each segment.
        iot_segm = segmScanExc (+) 0 segment_sizes (replicate n 1)
        
        -- Compute the new flags. At each position, loop over the equivalence
        -- class number (k = 0..m-1). If we are at the appropriate position, put
        -- the number of elements in class k at the position given by the offset
        -- for class k.
        new_flags =
            zipWith3 (\iot reduction class_offset ->
                      foldl (\acc k ->
                             let
                                 reduction_k = reduction !! k
                                 class_offset_k = class_offset !! k
                             in
                               if iot == class_offset_k && reduction_k > 0
                               -- We are at an appropriate position, and class
                               -- k actually contains elements.
                               then reduction_k
                               -- Otherwise, don't set the flag.
                               else acc
                            ) 0 [0..m-1]
                     )
            iot_segm reductions class_offsets
    in
      (new_flags, new_arr)
