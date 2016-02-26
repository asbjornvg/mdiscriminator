module Mdiscr4
    (
     mdiscr4
    ) where

import Helpers
import Data.List(zip5)

-- Up to four equivalence classes (not generalized).
mdiscr4 :: (a -> Int) -> [a] -> ([Int], [a])
mdiscr4 discr arr = (flags, permute inds arr)
    where
      n        = length arr
      ds       = map discr arr
      
      class0   = map (\f -> if f == 0 then 1 else 0) ds
      indices0 = scanInc (+) 0 class0
      i0       = last indices0
      
      class1   = map (\f -> if f == 1 then 1 else 0) ds
      indices1 = (map (+ i0) . scanInc (+) 0) class1
      i1       = last indices1
      
      class2   = map (\f -> if f == 2 then 1 else 0) ds
      indices2 = (map (+ i1) . scanInc (+) 0) class2
      i2       = last indices2
      
      class3   = map (\f -> if f == 3 then 1 else 0) ds
      indices3 = (map (+ i2) . scanInc (+) 0) class3
      i3       = last indices3
      
      inds     = map (\(d,is0,is1,is2,is3) ->
                          case d of 0 -> is0-1
                                    1 -> is1-1
                                    2 -> is2-1
                                    3 -> is3-1
                     )
                 (zip5 ds indices0 indices1 indices2 indices3)
      flags    = write [0,i0,i1,i2] [i0,i1-i0,i2-i1,i3-i2] (replicate n 0)
