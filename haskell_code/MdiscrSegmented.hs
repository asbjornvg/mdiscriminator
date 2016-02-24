module MdiscrSegmented
    (
     segmMdiscr
    ) where

import Helpers
import Data.List(zipWith4)

-- Segmented version.
--segmMdiscr :: Int -> (a -> Int) -> [a] -> [Int] -> ([a], [Int])
segmMdiscr m discr segment_sizes arr = new_arr
    where
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
      
      -- Create a list with [0,...,1,...,0] of length m where the 1 is in
      -- the k'th position.
      createColumn :: Int -> [Int]
      createColumn k =
          let
              zeros = replicate m 0
              (ys,_:zs) = splitAt k zeros
          in
            ys ++ (1 : zs)
      
      -- First turn the classes into columns (lists)
      columns = map createColumn classes
      
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
      
      -- Get the indices by adding the appropriate offsets and selecting the
      -- appropriate entries from the columns.
      indices = zipWith4 (\k scan_result class_offset segment_offset ->
                          -- Select the k'th entry from the column.
                          flip (!!) k $
                          -- Add the segment offsets to all the segments (and
                          -- make it 0-indexed).
                          map (+(segment_offset-1)) $
                          -- Add the class offsets (within segments).
                          zipWith (+) scan_result class_offset
                         )
                classes scan_results class_offsets segment_offsets
      
      -- Compute the resulting array based on the indices.
      new_arr = permute indices arr
      
      -- -- Shift the flags one to the left, so that now we are accessing the last
      -- -- elements of the segments.
      -- shifted_flags = tail segment_flags ++ [head segment_flags]
      
      -- aggr_stripped = zipWith (\flag scan_result -> if flag > 0
      --                                               then scan_result
      --                                               -- "Neutral element" column.
      --                                               else replicate m 0)
      --                 shifted_flags scan_results
      
      
      
      -- -- Add the offsets.
      -- scan_result_with_offsets = map (zipWith (+) offsets) scan_result
      
      -- -- Now, the indices for elements of class k can be found at the k'th entry
      -- -- in the corresponding column of the scan result, so we must extract the
      -- -- appropriate entries from the columns. These indices are 1-indexed, so
      -- -- we must remember to subtract 1.
      -- indices = zipWith (\k column -> column !! k - 1)
      --           classes scan_result_with_offsets
      
      -- -- Write a flag array containing the sizes (i.e., the number of elements
      -- -- belonging to each class) at the positions given by the offsets.
      -- flags = write offsets sizes (replicate n 0)
