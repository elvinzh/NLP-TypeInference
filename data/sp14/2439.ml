
let rec clone x n = if n <= 0 then [] else [x] @ (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then clone (l1, (0 :: l2))
  else
    if (List.length l1) < (List.length l2)
    then clone ((0 :: l1), l2)
    else (l1, l2);;


(* fix

let rec clone x n = if n <= 0 then [] else [x] @ (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then
    let l1G = (List.length l1) - (List.length l2) in
    (l1, (List.append (clone 0 l1G) l2))
  else
    if (List.length l1) < (List.length l2)
    then
      (let l2G = (List.length l2) - (List.length l1) in
       ((List.append (clone 0 l2G) l1), l2))
    else (l1, l2);;

*)

(* changed spans
(6,7)-(6,12)
(6,7)-(6,28)
(6,13)-(6,28)
(6,18)-(6,27)
(6,19)-(6,20)
(6,24)-(6,26)
(9,9)-(9,14)
(9,9)-(9,30)
(9,15)-(9,30)
(9,16)-(9,25)
(9,17)-(9,18)
(9,22)-(9,24)
*)

(* type error slice
(2,49)-(2,66)
(2,50)-(2,55)
(8,4)-(10,17)
(9,9)-(9,14)
(9,9)-(9,30)
(10,9)-(10,17)
*)

(* all spans
(2,14)-(2,66)
(2,16)-(2,66)
(2,20)-(2,66)
(2,23)-(2,29)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,37)
(2,43)-(2,66)
(2,47)-(2,48)
(2,43)-(2,46)
(2,44)-(2,45)
(2,49)-(2,66)
(2,50)-(2,55)
(2,56)-(2,57)
(2,58)-(2,65)
(2,59)-(2,60)
(2,63)-(2,64)
(4,12)-(10,17)
(4,15)-(10,17)
(5,2)-(10,17)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,28)
(6,7)-(6,12)
(6,13)-(6,28)
(6,14)-(6,16)
(6,18)-(6,27)
(6,19)-(6,20)
(6,24)-(6,26)
(8,4)-(10,17)
(8,7)-(8,42)
(8,7)-(8,23)
(8,8)-(8,19)
(8,20)-(8,22)
(8,26)-(8,42)
(8,27)-(8,38)
(8,39)-(8,41)
(9,9)-(9,30)
(9,9)-(9,14)
(9,15)-(9,30)
(9,16)-(9,25)
(9,17)-(9,18)
(9,22)-(9,24)
(9,27)-(9,29)
(10,9)-(10,17)
(10,10)-(10,12)
(10,14)-(10,16)
*)
