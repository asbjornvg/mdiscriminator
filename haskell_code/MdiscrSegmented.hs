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
        
        -- Get the segment boundaries as 1's.
        segment_flags = map (\s -> if s /= 0 then 1 else 0) segment_sizes
        
        -- Sizes extended inside each segment.
        sizes_extended = segmScanInc (+) 0 segment_flags segment_sizes
        
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
        scan_results = segmScanInc (zipWith (+)) (replicate m 0) segment_flags columns
        
        -- Distribute the last elements of the scan_results (i.e., the reductions)
        -- backwards across the segments.
        reductions = zipWith (\size_ext offset -> scan_results !! (size_ext+offset-1))
                     sizes_extended segment_offsets
        
        -- The reductions count the number of occurrences of each class. With the
        -- exclusive scan, we get the offsets that we must add to each class
        -- inside the segments.
        class_offsets = map (scanExc (+) 0) reductions
        
        -- Get the indices by selecting the appropriate entries from the columns
        -- and adding the appropriate offsets.
        indices = zipWith4 (\k scan_result class_offset segment_offset ->
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
        
        -- Shift the flags one to the left to make it a mask that accesses the
        -- last elements of the segments.
        shifted_flags = tail segment_flags ++ [head segment_flags]
        
        -- Mask out everyting but the last element of each segment. These
        -- elements contain the reductions of the segments.
        reductions_stripped = zipWith (\flag scan_result ->
                                       if flag > 0
                                       -- The last element of the segment
                                       -- contains the reduction.
                                       then scan_result
                                       -- "Neutral element" column.
                                       else replicate m 0)
                              shifted_flags scan_results
        
        -- Loop over the class number (for k = 0..m-1). We start with a flag
        -- array of zeros of length (n+1), and for each class k, we will write
        -- the size of equivalence class k (in each segment) to the appropriate
        -- place in the flag array. The 0'th element contains only "junk" and
        -- is discarded (by taking only the tail).
        new_flags = tail $
                    foldl (\acc_sizes k ->
                           let
                               -- Here, we are producing [(index,size)], and
                               -- then we unzip to make it ([indices],[sizes]).
                               (indices_k, sizes_k) =
                                   unzip $
                                   zipWith3
                                   -- For the current class k, extract the
                                   -- k'th column from the masked out array
                                   -- of reductions.
                                   (\reduction segment_offset class_offset ->
                                    let
                                        reduction_k = reduction !! k
                                        class_offset_k = class_offset !! k
                                    in
                                      if reduction_k > 0
                                      -- Only if there actually were some
                                      -- occurrences for this class do we write
                                      -- into the flag array.
                                      then (segment_offset + class_offset_k + 1,
                                            reduction_k)
                                      -- Otherwise, either there were no
                                      -- occurrences, or we are not at a
                                      -- reduction (i.e., we are currently at
                                      -- an element that was masked out). Either
                                      -- way, we simply don't write to the flag
                                      -- array.
                                      else (0,0) -- junk
                                   )
                                   reductions_stripped
                                   segment_offsets
                                   class_offsets
                           in
                             write indices_k sizes_k acc_sizes
                          )
                    -- We start with a flag array of zeros of length (n+1).
                    (replicate (n+1) 0)
                    -- Loop over the class number (for k = 0..m-1).
                    [0..m-1]
    in
      (new_flags, new_arr)
