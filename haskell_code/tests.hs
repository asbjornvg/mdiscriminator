import Mdiscr(mdiscr)
import Mdiscr4(mdiscr4)

arr :: [Int]
arr = [5,4,2,3,7,8,6,4,1,9,11,12,10]

test :: Int -> IO ()
test m = do
  let discr x = x `mod` m
  putStrLn "\t- mdiscr"
  putStrLn $ (++) "\t  " $ show $ mdiscr m discr arr
  putStrLn "\t- mdiscr4"
  putStrLn $ (++) "\t  " $ show $ mdiscr4 discr arr

test_only_generic :: Int -> IO ()
test_only_generic m = do
  let discr x = x `mod` m
  putStrLn "\t- mdiscr"
  putStrLn $ (++) "\t  " $ show $ mdiscr m discr arr

main :: IO ()
main = do
  putStrLn "Original array:"
  putStrLn $ (++) "\t" $ show arr
  putStrLn "Five equivalence classes:"
  test_only_generic 5
  putStrLn "Four equivalence classes:"
  test 4
  putStrLn "Three equivalence classes:"
  test 3
  putStrLn "Two equivalence classes:"
  test 2
  putStrLn "One equivalence class:"
  test 1
