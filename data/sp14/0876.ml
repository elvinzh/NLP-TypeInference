
let rec clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2 then (((clone 0 (len1 - len2)) @ len2), len1);;


(* fix

let rec clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2
  then (((clone 0 (len1 - len2)) @ l2), l1)
  else (((clone 0 (len2 - len1)) @ l1), l2);;

*)

(* changed spans
(10,2)-(10,62)
(10,50)-(10,54)
*)

(* type error slice
(9,2)-(10,62)
(9,13)-(9,24)
(9,13)-(9,27)
(10,2)-(10,62)
(10,22)-(10,62)
(10,23)-(10,55)
(10,48)-(10,49)
(10,50)-(10,54)
*)

(* all spans
(2,14)-(5,15)
(2,16)-(5,15)
(3,2)-(5,15)
(3,17)-(4,55)
(3,19)-(4,55)
(3,21)-(4,55)
(4,4)-(4,55)
(4,7)-(4,13)
(4,7)-(4,8)
(4,12)-(4,13)
(4,19)-(4,22)
(4,28)-(4,55)
(4,28)-(4,34)
(4,35)-(4,36)
(4,37)-(4,44)
(4,38)-(4,39)
(4,42)-(4,43)
(4,45)-(4,55)
(4,46)-(4,47)
(4,51)-(4,54)
(5,2)-(5,15)
(5,2)-(5,8)
(5,9)-(5,10)
(5,11)-(5,12)
(5,13)-(5,15)
(7,12)-(10,62)
(7,15)-(10,62)
(8,2)-(10,62)
(8,13)-(8,27)
(8,13)-(8,24)
(8,25)-(8,27)
(9,2)-(10,62)
(9,13)-(9,27)
(9,13)-(9,24)
(9,25)-(9,27)
(10,2)-(10,62)
(10,5)-(10,16)
(10,5)-(10,9)
(10,12)-(10,16)
(10,22)-(10,62)
(10,23)-(10,55)
(10,48)-(10,49)
(10,24)-(10,47)
(10,25)-(10,30)
(10,31)-(10,32)
(10,33)-(10,46)
(10,34)-(10,38)
(10,41)-(10,45)
(10,50)-(10,54)
(10,57)-(10,61)
*)
