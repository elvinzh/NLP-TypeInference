
let rec wwhile (f,b) =
  match f b with | (i,true ) -> wwhile (f, i) | (i,false ) -> i;;

let fixpoint (f,b) =
  wwhile (fun x  -> if x = (f x) then (x, false) else (((f x), true), b));;


(* fix

let rec wwhile (f,b) =
  match f b with | (i,true ) -> wwhile (f, i) | (i,false ) -> i;;

let fixpoint (f,b) =
  wwhile ((fun x  -> if x = (f x) then (x, false) else ((f x), true)), b);;

*)

(* changed spans
(6,9)-(6,73)
(6,55)-(6,68)
*)

(* type error slice
(6,20)-(6,72)
(6,23)-(6,24)
(6,23)-(6,32)
(6,27)-(6,32)
(6,28)-(6,29)
(6,38)-(6,48)
(6,39)-(6,40)
(6,54)-(6,72)
(6,55)-(6,68)
(6,56)-(6,61)
(6,57)-(6,58)
*)

(* all spans
(2,16)-(3,63)
(3,2)-(3,63)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(3,32)-(3,45)
(3,32)-(3,38)
(3,39)-(3,45)
(3,40)-(3,41)
(3,43)-(3,44)
(3,62)-(3,63)
(5,14)-(6,73)
(6,2)-(6,73)
(6,2)-(6,8)
(6,9)-(6,73)
(6,20)-(6,72)
(6,23)-(6,32)
(6,23)-(6,24)
(6,27)-(6,32)
(6,28)-(6,29)
(6,30)-(6,31)
(6,38)-(6,48)
(6,39)-(6,40)
(6,42)-(6,47)
(6,54)-(6,72)
(6,55)-(6,68)
(6,56)-(6,61)
(6,57)-(6,58)
(6,59)-(6,60)
(6,63)-(6,67)
(6,70)-(6,71)
*)
