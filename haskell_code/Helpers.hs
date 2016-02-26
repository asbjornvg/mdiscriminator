module Helpers
    (
     split
    ,reduce
    ,red
    ,iota
    ,scanInc
    ,scanExc
    ,segmScanInc
    ,segmScanExc
    ,permute
    ,write
    ,createColumn
    ) where

import Data.List(sortBy)
import Data.Function(on)

split :: [a] -> ([a], [a])
split []  = ([],[] )
split [x] = ([],[x]) 
split xs  = let mid = (length xs) `div` 2
            in (take mid xs, drop mid xs)

reduce :: (a -> a -> a) -> a -> [a] -> a
reduce = foldl

red :: (a -> a -> a) -> a -> [a] -> a
red op e []  = e
red op e [x] = x
red op e xs  = let (x, y) = split xs
               in  (red op e x) `op` (red op e y)

iota :: Int -> [Int]
iota n = [0..n-1]

scanInc :: (a -> a -> a) -> a -> [a] -> [a]
scanInc op e xs = drop 1 $ scanl op e xs

scanExc :: (a -> a -> a) -> a -> [a] -> [a]
scanExc _ _ []  = []
scanExc op e xs = take (length xs) $ scanl op e xs

segmScanInc :: (a -> a -> a) -> a -> [Int] -> [a] -> [a]
segmScanInc myop ne flags arr =
    tail $
         scanl (\v1 (f,v2) -> 
                if f == 0 
                then v1 `myop` v2
                else ne `myop` v2
               ) ne $ zip flags arr

segmScanExc :: (a -> a -> a) -> a -> [Int] -> [a] -> [a]
segmScanExc myop ne flags arr =
    let inds   = iota (length arr)
        adj_arr= zipWith (\i f -> if f>0 then ne 
                                  else (arr !! (i-1)))
                         inds flags
    in  segmScanInc myop ne flags adj_arr

permute :: [Int] -> [a] -> [a]
permute indices arr =
    snd . unzip . sortBy (compare `on` fst) $ zip indices arr

write :: [Int] -> [a] -> [a] -> [a]
write indices elements arr =
    snd . unzip $ aux original_elements new_elements
    where
      new_elements = zip indices elements
      original_elements = zip [0..] arr
      aux :: [(Int,a)] -> [(Int,a)] -> [(Int,a)]
      aux [] _      = []
      aux xs []     = xs -- Until list of y's is empty
      aux xs (y:ys) = aux (map (replaceWith y) xs) ys
      replaceWith :: (Int,a) -> (Int,a) -> (Int,a)
      replaceWith y x = if (fst y) == (fst x)
                        then y -- Indices matched, so replace
                        else x -- Keep original

-- Create a list with [0,...,1,...,0] of length m where the 1 is in
-- the k'th position.
createColumn :: Int -> Int -> [Int]
createColumn m k =
    map (\position -> if position == k
                      then 1
                      else 0) $ iota m

-- Another way of doing it.
createColumn2 :: Int -> Int -> [Int]
createColumn2 m k =
    let
        zeros = replicate m 0
        (ys,_:zs) = splitAt k zeros
    in
      ys ++ (1 : zs)
