
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) = (List.length l2)
  then [(l1, l2)]
  else
    (let numZeros = (List.length l1) - (List.length l2) in
     if numZeros = 0
     then [(l1, l2)]
     else
       (let listZeros = clone (0, (abs numZeros)) in
        if numZeros > 0
        then let list1 = l1 in let list2 = listZeros @ l2 in [(list1, list2)]
        else
          (let list1 = listZeros @ l1 in let list2 = l2 in [(list1, list2)])));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let numZeros = (List.length l1) - (List.length l2) in
  let absNumZeros = abs numZeros in
  if numZeros = 0
  then (l1, l2)
  else
    (let listZeros = clone 0 absNumZeros in
     if numZeros > 0 then (l1, (listZeros @ l2)) else ((listZeros @ l1), l2));;

*)

(* changed spans
(5,2)-(16,78)
(5,5)-(5,21)
(5,5)-(5,40)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,17)
(6,8)-(6,16)
(6,9)-(6,11)
(6,13)-(6,15)
(9,5)-(16,77)
(10,10)-(10,20)
(12,24)-(12,49)
(12,30)-(12,49)
(12,34)-(12,48)
(12,35)-(12,38)
(12,39)-(12,47)
(14,13)-(14,77)
(14,31)-(14,77)
(14,61)-(14,77)
(14,63)-(14,68)
(14,70)-(14,75)
(16,10)-(16,76)
(16,41)-(16,75)
(16,59)-(16,75)
(16,60)-(16,74)
(16,61)-(16,66)
(16,68)-(16,73)
*)

(* type error slice
(2,48)-(2,65)
(2,49)-(2,54)
(12,7)-(16,77)
(12,24)-(12,29)
(12,24)-(12,49)
(14,43)-(14,52)
(14,43)-(14,57)
(14,53)-(14,54)
(16,23)-(16,32)
(16,23)-(16,37)
(16,33)-(16,34)
*)

(* all spans
(2,14)-(2,65)
(2,16)-(2,65)
(2,20)-(2,65)
(2,23)-(2,29)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,37)
(2,43)-(2,65)
(2,43)-(2,44)
(2,48)-(2,65)
(2,49)-(2,54)
(2,55)-(2,56)
(2,57)-(2,64)
(2,58)-(2,59)
(2,62)-(2,63)
(4,12)-(16,78)
(4,15)-(16,78)
(5,2)-(16,78)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,17)
(6,8)-(6,16)
(6,9)-(6,11)
(6,13)-(6,15)
(8,4)-(16,78)
(8,20)-(8,55)
(8,20)-(8,36)
(8,21)-(8,32)
(8,33)-(8,35)
(8,39)-(8,55)
(8,40)-(8,51)
(8,52)-(8,54)
(9,5)-(16,77)
(9,8)-(9,20)
(9,8)-(9,16)
(9,19)-(9,20)
(10,10)-(10,20)
(10,11)-(10,19)
(10,12)-(10,14)
(10,16)-(10,18)
(12,7)-(16,77)
(12,24)-(12,49)
(12,24)-(12,29)
(12,30)-(12,49)
(12,31)-(12,32)
(12,34)-(12,48)
(12,35)-(12,38)
(12,39)-(12,47)
(13,8)-(16,76)
(13,11)-(13,23)
(13,11)-(13,19)
(13,22)-(13,23)
(14,13)-(14,77)
(14,25)-(14,27)
(14,31)-(14,77)
(14,43)-(14,57)
(14,53)-(14,54)
(14,43)-(14,52)
(14,55)-(14,57)
(14,61)-(14,77)
(14,62)-(14,76)
(14,63)-(14,68)
(14,70)-(14,75)
(16,10)-(16,76)
(16,23)-(16,37)
(16,33)-(16,34)
(16,23)-(16,32)
(16,35)-(16,37)
(16,41)-(16,75)
(16,53)-(16,55)
(16,59)-(16,75)
(16,60)-(16,74)
(16,61)-(16,66)
(16,68)-(16,73)
*)
