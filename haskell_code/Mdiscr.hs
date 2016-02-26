module Mdiscr
    (
     mdiscr
    ) where

import Helpers

-- Generalized version with a single scan.
mdiscr :: Int -> (a -> Int) -> [a] -> ([Int], [a])
mdiscr m discr arr =
    let
        -- The length of the input array.
        n = length arr
        
        -- Find the equivalence classes using the discriminator.
        classes = map discr arr
        
        -- Here, we could assert that all the classes are between 0 and m-1.
        
        -- Turn the classes into columns (lists).
        columns = map (createColumn m) classes
        
        -- Scan the columns.
        scan_results = scanInc (zipWith (+)) (replicate m 0) columns
        
        -- The last column contains the reductions for each class, i.e., the
        -- total number of elements belonging to each class.
        sizes = last scan_results
        
        -- Since all the elements of class k come before the elements of class
        -- k+1, we must accumulate the sizes into offsets to add to each class.
        offsets = scanExc (+) 0 sizes
        
        -- Get the indices by selecting the appropriate entry from the scanned
        -- column and adding the corresponding offset.
        indices = zipWith (\k scan_result ->
                           let
                               -- Select the k'th entries.
                               scan_result_k = scan_result !! k
                               offset_k = offsets !! k
                           in
                             -- Add offset. Subtract 1 to make it 0-indexed.
                             scan_result_k + offset_k - 1
                          )
                  classes scan_results
        
        -- Compute the resulting array based on the indices.
        new_arr = permute indices arr
        
        -- Write a flag array containing the sizes (i.e., the number of elements
        -- belonging to each class) at the positions given by the offsets.
        new_flags = write offsets sizes (replicate n 0)
    in
      (new_flags, new_arr)
