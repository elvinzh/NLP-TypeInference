
let rec clone x n = let accum = [] in if n < 1 then [] else (clone x n) - 1;;


(* fix

let rec clone x n = let accum = [] in if n < 1 then [] else clone x (n - 1);;

*)

(* changed spans
(2,60)-(2,75)
(2,69)-(2,70)
*)

(* type error slice
(2,38)-(2,75)
(2,52)-(2,54)
(2,60)-(2,75)
*)

(* all spans
(2,14)-(2,75)
(2,16)-(2,75)
(2,20)-(2,75)
(2,32)-(2,34)
(2,38)-(2,75)
(2,41)-(2,46)
(2,41)-(2,42)
(2,45)-(2,46)
(2,52)-(2,54)
(2,60)-(2,75)
(2,60)-(2,71)
(2,61)-(2,66)
(2,67)-(2,68)
(2,69)-(2,70)
(2,74)-(2,75)
*)