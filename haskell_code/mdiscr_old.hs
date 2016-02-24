module Mdiscr
    (
     mdiscr
    ) where

import Helpers

-- Generalized version.
mdiscr :: Int -> (a -> Int) -> [a] -> ([a], [Int])
mdiscr m discr arr = (permute inds arr, flags)
    where
      -- The length of the input array.
      n = length arr
      
      -- Find the equivalence classes using the discriminator.
      ds = map discr arr
      
      -- The fold corresponds to a sequential for-loop (for k = 0 .. m-1),
      -- because in each iteration we need the accumulated value of the
      -- previous iteration.
      (inds, offsets, _) =
          foldl (\(acc_list, acc_offset_list, current_offset) k ->
                 let
                     -- Inside the sequential loop, there are some maps, scans
                     -- and zips (i.e., parallel inner loops).
                     
                     -- Flags that say which elements belong to the current
                     -- equivalence class k.
                     flags = map (\d -> if d==k then 1 else 0) ds
                     
                     -- Accumulate the flags with a scan and add the offset
                     -- from the last class to get the indices for this class
                     -- (1-indexed).
                     indices = (map (+ current_offset) . scanInc (+) 0) flags
                     
                     -- The last index of class k is the offset to class k+1.
                     new_offset = last indices
                     
                     -- Merge the indices for this class into the accumulating
                     -- list of indices.
                     result = zipWith3 (\current flag index ->
                                        if flag == 1
                                        then index-1 -- Make it 0-indexed
                                        else current
                                       ) acc_list flags indices
                 in
                   -- Pass on the merged incides, add the new offset to the list
                   -- of offsets, and pass on the new offset.
                   (result, acc_offset_list ++ [new_offset], new_offset)
                   
                -- Start with an accumulated list of all zeros, an empty list of
                -- offsets, and a starting offset of 0.
                ) (replicate n 0, [], 0) [0..m-1]
      
      -- Write the flags (containing the lengths of the classes) into an array
      -- of all zeros.
      flags = write
              -- Shift the offsets right by one.
              (0 : take (length offsets - 1) offsets)
              -- [offset_0 - 0, offset_1 - offset_0, ..., offset_(m-1) - offset_(m-2)]
              (snd $ foldl (\(prev,acc) elem -> (elem,acc++[elem-prev])) (0,[]) offsets)
              -- All zeros.
              (replicate n 0)
      
      -- flags = write [0,i0,i1,i2] [i0,i1-i0,i2-i1,i3-i2] (replicate n 0)
