
let rec clone x n = match n with | 0 -> [] | _ -> [clone x (n - 1); x];;


(* fix

let rec clone x n = match n with | 0 -> [] | _ -> (clone x (n - 1)) @ [x];;

*)

(* changed spans
(2,50)-(2,70)
(2,51)-(2,56)
(2,68)-(2,69)
*)

(* type error slice
(2,3)-(2,72)
(2,14)-(2,70)
(2,16)-(2,70)
(2,20)-(2,70)
(2,50)-(2,70)
(2,51)-(2,56)
(2,51)-(2,66)
*)

(* all spans
(2,14)-(2,70)
(2,16)-(2,70)
(2,20)-(2,70)
(2,26)-(2,27)
(2,40)-(2,42)
(2,50)-(2,70)
(2,51)-(2,66)
(2,51)-(2,56)
(2,57)-(2,58)
(2,59)-(2,66)
(2,60)-(2,61)
(2,64)-(2,65)
(2,68)-(2,69)
*)
