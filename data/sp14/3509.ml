
let rec wwhile (f,b) =
  match f b with
  | (b',c') -> (match c' with | true  -> wwhile (f, b') | false  -> b');;

let fixpoint (f,b) = wwhile (((not f b) b), b);;


(* fix

let rec wwhile (f,b) =
  match f b with
  | (b',c') -> (match c' with | true  -> wwhile (f, b') | false  -> b');;

let fixpoint (f,b) = wwhile (let func x x = (0, true) in ((func b), b));;

*)

(* changed spans
(6,28)-(6,46)
(6,29)-(6,42)
(6,30)-(6,39)
(6,31)-(6,34)
(6,35)-(6,36)
(6,37)-(6,38)
*)

(* type error slice
(6,30)-(6,39)
(6,31)-(6,34)
*)

(* all spans
(2,16)-(4,71)
(3,2)-(4,71)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(4,15)-(4,71)
(4,22)-(4,24)
(4,41)-(4,55)
(4,41)-(4,47)
(4,48)-(4,55)
(4,49)-(4,50)
(4,52)-(4,54)
(4,68)-(4,70)
(6,14)-(6,46)
(6,21)-(6,46)
(6,21)-(6,27)
(6,28)-(6,46)
(6,29)-(6,42)
(6,30)-(6,39)
(6,31)-(6,34)
(6,35)-(6,36)
(6,37)-(6,38)
(6,40)-(6,41)
(6,44)-(6,45)
*)
