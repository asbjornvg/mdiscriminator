module Mdiscr
    (
     mdiscr
    ) where

import Helpers

-- Generalized version with a single scan.
mdiscr :: Int -> (a -> Int) -> [a] -> ([a], [Int])
mdiscr m discr arr = (new_arr, sizes)
    where
      -- The length of the input array.
      n = length arr
      
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
      
      -- First turn the classes into columns (lists), then scan. No offsets are
      -- added yet.
      scan_result = (scanInc (zipWith (+)) (replicate m 0) .
                     map createColumn) classes
      
      -- The last column contains the reductions for each class, i.e., the
      -- total number of elements belonging to each class.
      sizes = last scan_result
      
      -- Since all the elements of class k come before the elements of class
      -- k+1, we must accumulate the sizes into offsets to add to each class.
      offsets = scanExc (+) 0 sizes
      
      -- Add the offsets.
      scan_result_with_offsets = map (zipWith (+) offsets) scan_result
      
      -- Now, the indices for elements of class k can be found at the k'th entry
      -- in the corresponding column of the scan result, so we must extract the
      -- appropriate entries from the columns. These indices are 1-indexed, so
      -- we must remember to subtract 1.
      indices = zipWith (\k column -> column !! k - 1)
                classes scan_result_with_offsets
      
      new_arr = permute indices arr
      
      -- -- Write a flag array containing the sizes (i.e., the number of elements
      -- -- belonging to each class) at the positions given by the offsets.
      -- flags = write offsets sizes (replicate n 0)
