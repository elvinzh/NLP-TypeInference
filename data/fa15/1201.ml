
let rec wwhile (f,b) = match f b with | (h,t) -> if t = false then f;;


(* fix

let rec wwhile (f,b) = match f b with | (h,t) -> if t = false then h;;

*)

(* changed spans
(2,67)-(2,68)
*)

(* type error slice
(2,29)-(2,30)
(2,29)-(2,32)
(2,49)-(2,68)
(2,67)-(2,68)
*)

(* all spans
(2,16)-(2,68)
(2,23)-(2,68)
(2,29)-(2,32)
(2,29)-(2,30)
(2,31)-(2,32)
(2,49)-(2,68)
(2,52)-(2,61)
(2,52)-(2,53)
(2,56)-(2,61)
(2,67)-(2,68)
*)
