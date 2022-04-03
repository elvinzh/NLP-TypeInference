
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let numZeros = (List.length l1) - (List.length l2) in
  let listZeros = clone 0 abs numZeros in
  if numZeros > 0 then [(l1, (listZeros @ l2))] else [((listZeros @ l1), l2)];;


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
(6,2)-(7,77)
(6,18)-(6,38)
(6,26)-(6,29)
(6,30)-(6,38)
(7,23)-(7,47)
(7,53)-(7,77)
*)

(* type error slice
(2,48)-(2,65)
(2,49)-(2,54)
(2,57)-(2,64)
(6,18)-(6,23)
(6,18)-(6,38)
(6,26)-(6,29)
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
(4,12)-(7,77)
(4,15)-(7,77)
(5,2)-(7,77)
(5,17)-(5,52)
(5,17)-(5,33)
(5,18)-(5,29)
(5,30)-(5,32)
(5,36)-(5,52)
(5,37)-(5,48)
(5,49)-(5,51)
(6,2)-(7,77)
(6,18)-(6,38)
(6,18)-(6,23)
(6,24)-(6,25)
(6,26)-(6,29)
(6,30)-(6,38)
(7,2)-(7,77)
(7,5)-(7,17)
(7,5)-(7,13)
(7,16)-(7,17)
(7,23)-(7,47)
(7,24)-(7,46)
(7,25)-(7,27)
(7,29)-(7,45)
(7,40)-(7,41)
(7,30)-(7,39)
(7,42)-(7,44)
(7,53)-(7,77)
(7,54)-(7,76)
(7,55)-(7,71)
(7,66)-(7,67)
(7,56)-(7,65)
(7,68)-(7,70)
(7,73)-(7,75)
*)
