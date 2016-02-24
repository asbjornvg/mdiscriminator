import MdiscrSegmented(segmMdiscr)

arr :: [Int]
arr = [5,4,2,3,7,8,6,4,1,9,11,12,10]

sizes :: [Int]
sizes = [4,0,0,0,2,0,1,3,0,0,3,0,0]

test :: Int -> IO ()
test m = do
  let discr x = x `mod` m
  putStrLn "\t- segmMdiscr"
  putStrLn $ (++) "\t  " $ show $ segmMdiscr m discr sizes arr

main :: IO ()
main = do
  putStrLn "Sizes and original array:"
  putStrLn ("\t" ++ show sizes ++ "\n\t" ++ show arr)
  putStrLn "Four equivalence classes:"
  test 4
