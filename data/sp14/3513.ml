
let rec wwhile (f,b) =
  match f b with
  | (b',c') -> (match c' with | true  -> wwhile (f, b') | false  -> b');;

let fixpoint (f,b) = wwhile (fun f  -> fun b  -> ((f, ((f b) <> b)), b));;


(* fix

let rec wwhile (f,b) =
  match f b with
  | (b',c') -> (match c' with | true  -> wwhile (f, b') | false  -> b');;

let fixpoint (f,b) = wwhile (let func x x = (0, true) in ((func b), b));;

*)

(* changed spans
(6,28)-(6,72)
(6,39)-(6,71)
(6,49)-(6,71)
(6,50)-(6,67)
(6,51)-(6,52)
(6,54)-(6,66)
(6,56)-(6,57)
(6,58)-(6,59)
*)

(* type error slice
(4,41)-(4,47)
(4,41)-(4,55)
(4,48)-(4,55)
(6,21)-(6,27)
(6,21)-(6,72)
(6,28)-(6,72)
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
(6,14)-(6,72)
(6,21)-(6,72)
(6,21)-(6,27)
(6,28)-(6,72)
(6,39)-(6,71)
(6,49)-(6,71)
(6,50)-(6,67)
(6,51)-(6,52)
(6,54)-(6,66)
(6,55)-(6,60)
(6,56)-(6,57)
(6,58)-(6,59)
(6,64)-(6,65)
(6,69)-(6,70)
*)
