
let rec wwhile (f,b) =
  let pair = f b in
  let (b',c') = pair in if c' = false then b' else wwhile (f, b');;

let fixpoint (f,b) = wwhile (true, b);;


(* fix

let rec wwhile (f,b) =
  let pair = f b in
  let (b',c') = pair in if c' = false then b' else wwhile (f, b');;

let fixpoint (f,b) = wwhile ((fun x  -> ((f x), (x != (f x)))), b);;

*)

(* changed spans
(6,29)-(6,33)
(6,35)-(6,36)
*)

(* type error slice
(3,13)-(3,14)
(3,13)-(3,16)
(4,51)-(4,57)
(4,51)-(4,65)
(4,58)-(4,65)
(4,59)-(4,60)
(6,21)-(6,27)
(6,21)-(6,37)
(6,28)-(6,37)
(6,29)-(6,33)
*)

(* all spans
(2,16)-(4,65)
(3,2)-(4,65)
(3,13)-(3,16)
(3,13)-(3,14)
(3,15)-(3,16)
(4,2)-(4,65)
(4,16)-(4,20)
(4,24)-(4,65)
(4,27)-(4,37)
(4,27)-(4,29)
(4,32)-(4,37)
(4,43)-(4,45)
(4,51)-(4,65)
(4,51)-(4,57)
(4,58)-(4,65)
(4,59)-(4,60)
(4,62)-(4,64)
(6,14)-(6,37)
(6,21)-(6,37)
(6,21)-(6,27)
(6,28)-(6,37)
(6,29)-(6,33)
(6,35)-(6,36)
*)
