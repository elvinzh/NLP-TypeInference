
let rec clone x n = if n < 1 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let difference1 = (List.length l1) - (List.length l2) in
  let difference2 = (List.length l2) - (List.length l1) in
  if difference2 > 0
  then ((l1 :: (clone 0 difference2)), l2)
  else
    if difference1 > 0 then (l1, (l2 :: (clone 0 difference1))) else (l1, l2);;


(* fix

let rec clone x n = if n < 1 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let difference1 = (List.length l1) - (List.length l2) in
  let difference2 = (List.length l2) - (List.length l1) in
  if difference2 > 0
  then (((clone 0 difference2) @ l1), l2)
  else
    if difference1 > 0 then (l1, ((clone 0 difference1) @ l2)) else (l1, l2);;

*)

(* changed spans
(8,8)-(8,37)
(8,9)-(8,11)
(8,16)-(8,21)
(8,39)-(8,41)
(10,33)-(10,62)
(10,34)-(10,36)
(10,41)-(10,46)
(10,69)-(10,77)
*)

(* type error slice
(2,42)-(2,43)
(2,42)-(2,64)
(2,47)-(2,64)
(2,48)-(2,53)
(2,54)-(2,55)
(6,20)-(6,36)
(6,21)-(6,32)
(6,33)-(6,35)
(6,39)-(6,55)
(6,40)-(6,51)
(6,52)-(6,54)
(8,8)-(8,37)
(8,9)-(8,11)
(8,15)-(8,36)
(8,16)-(8,21)
(8,22)-(8,23)
(10,4)-(10,77)
(10,28)-(10,63)
(10,33)-(10,62)
(10,34)-(10,36)
(10,40)-(10,61)
(10,41)-(10,46)
(10,69)-(10,77)
(10,74)-(10,76)
*)

(* all spans
(2,14)-(2,64)
(2,16)-(2,64)
(2,20)-(2,64)
(2,23)-(2,28)
(2,23)-(2,24)
(2,27)-(2,28)
(2,34)-(2,36)
(2,42)-(2,64)
(2,42)-(2,43)
(2,47)-(2,64)
(2,48)-(2,53)
(2,54)-(2,55)
(2,56)-(2,63)
(2,57)-(2,58)
(2,61)-(2,62)
(4,12)-(10,77)
(4,15)-(10,77)
(5,2)-(10,77)
(5,20)-(5,55)
(5,20)-(5,36)
(5,21)-(5,32)
(5,33)-(5,35)
(5,39)-(5,55)
(5,40)-(5,51)
(5,52)-(5,54)
(6,2)-(10,77)
(6,20)-(6,55)
(6,20)-(6,36)
(6,21)-(6,32)
(6,33)-(6,35)
(6,39)-(6,55)
(6,40)-(6,51)
(6,52)-(6,54)
(7,2)-(10,77)
(7,5)-(7,20)
(7,5)-(7,16)
(7,19)-(7,20)
(8,7)-(8,42)
(8,8)-(8,37)
(8,9)-(8,11)
(8,15)-(8,36)
(8,16)-(8,21)
(8,22)-(8,23)
(8,24)-(8,35)
(8,39)-(8,41)
(10,4)-(10,77)
(10,7)-(10,22)
(10,7)-(10,18)
(10,21)-(10,22)
(10,28)-(10,63)
(10,29)-(10,31)
(10,33)-(10,62)
(10,34)-(10,36)
(10,40)-(10,61)
(10,41)-(10,46)
(10,47)-(10,48)
(10,49)-(10,60)
(10,69)-(10,77)
(10,70)-(10,72)
(10,74)-(10,76)
*)