
let rec clone x n = match n with | 0 -> [] | h::t -> x :: ((clone x n) - 1);;


(* fix

let rec clone x n = if n < 1 then [] else x :: (clone x (n - 1));;

*)

(* changed spans
(2,20)-(2,75)
(2,26)-(2,27)
(2,40)-(2,42)
(2,58)-(2,75)
(2,68)-(2,69)
*)

(* type error slice
(2,3)-(2,77)
(2,14)-(2,75)
(2,16)-(2,75)
(2,20)-(2,75)
(2,53)-(2,75)
(2,58)-(2,75)
(2,59)-(2,70)
(2,60)-(2,65)
*)

(* all spans
(2,14)-(2,75)
(2,16)-(2,75)
(2,20)-(2,75)
(2,26)-(2,27)
(2,40)-(2,42)
(2,53)-(2,75)
(2,53)-(2,54)
(2,58)-(2,75)
(2,59)-(2,70)
(2,60)-(2,65)
(2,66)-(2,67)
(2,68)-(2,69)
(2,73)-(2,74)
*)
