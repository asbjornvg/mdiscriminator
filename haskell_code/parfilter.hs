module Parfilter
    (
     parFilter
    ) where

import Helpers

parFilter :: (a -> Bool) -> [a] -> ([a], [Int])
parFilter cond arr = (permute inds arr, flags)
    where
      n     = length arr
      cs    = map cond arr
      tfs   = map (\f -> if f then 1 else 0) cs
      isT   = scanInc (+) 0 tfs
      i     = last isT
      ffs   = map (\f -> if f then 0 else 1) cs
      isF   = (map (+ i) . scanInc (+) 0) ffs
      inds  = map (\(c,iT,iF) ->
                   if c then iT-1 else iF-1)
              (zip3 cs isT isF)
      flags = write [0,i] [i,n-i] (replicate n 0)
