
let rec clone x n =
  match n with | 0 -> [] | n -> if n < 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  match (List.length l1) - (List.length l2) with
  | 0 -> (l1, l2)
  | n -> if n < 0 then (((clone 0 n) @ l1), l2);;


(* fix

let rec clone x n =
  match n with | 0 -> [] | n -> if n < 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  match (List.length l1) - (List.length l2) with
  | 0 -> (l1, l2)
  | n -> if n < 0 then (((clone 0 n) @ l1), l2) else (((clone 0 n) @ l2), l1);;

*)

(* changed spans
(8,9)-(8,47)
*)

(* type error slice
(8,9)-(8,47)
(8,23)-(8,47)
*)

(* all spans
(2,14)-(3,76)
(2,16)-(3,76)
(3,2)-(3,76)
(3,8)-(3,9)
(3,22)-(3,24)
(3,32)-(3,76)
(3,35)-(3,40)
(3,35)-(3,36)
(3,39)-(3,40)
(3,46)-(3,48)
(3,54)-(3,76)
(3,54)-(3,55)
(3,59)-(3,76)
(3,60)-(3,65)
(3,66)-(3,67)
(3,68)-(3,75)
(3,69)-(3,70)
(3,73)-(3,74)
(5,12)-(8,47)
(5,15)-(8,47)
(6,2)-(8,47)
(6,8)-(6,43)
(6,8)-(6,24)
(6,9)-(6,20)
(6,21)-(6,23)
(6,27)-(6,43)
(6,28)-(6,39)
(6,40)-(6,42)
(7,9)-(7,17)
(7,10)-(7,12)
(7,14)-(7,16)
(8,9)-(8,47)
(8,12)-(8,17)
(8,12)-(8,13)
(8,16)-(8,17)
(8,23)-(8,47)
(8,24)-(8,42)
(8,37)-(8,38)
(8,25)-(8,36)
(8,26)-(8,31)
(8,32)-(8,33)
(8,34)-(8,35)
(8,39)-(8,41)
(8,44)-(8,46)
*)
