
let rec wwhile (f,b) = let (x,y) = f b in if y then wwhile (f, x) else x;;

let fixpoint (f,b) = wwhile (fun x  -> (((f b), (not (b = (f b)))), b));;


(* fix

let rec wwhile (f,b) = let (x,y) = f b in if y then wwhile (f, x) else x;;

let fixpoint (f,b) = wwhile ((fun x  -> ((f b), (not (b = (f b))))), b);;

*)

(* changed spans
(4,28)-(4,71)
(4,40)-(4,66)
*)

(* type error slice
(2,52)-(2,58)
(2,52)-(2,65)
(2,59)-(2,65)
(4,21)-(4,27)
(4,21)-(4,71)
(4,28)-(4,71)
*)

(* all spans
(2,16)-(2,72)
(2,23)-(2,72)
(2,35)-(2,38)
(2,35)-(2,36)
(2,37)-(2,38)
(2,42)-(2,72)
(2,45)-(2,46)
(2,52)-(2,65)
(2,52)-(2,58)
(2,59)-(2,65)
(2,60)-(2,61)
(2,63)-(2,64)
(2,71)-(2,72)
(4,14)-(4,71)
(4,21)-(4,71)
(4,21)-(4,27)
(4,28)-(4,71)
(4,39)-(4,70)
(4,40)-(4,66)
(4,41)-(4,46)
(4,42)-(4,43)
(4,44)-(4,45)
(4,48)-(4,65)
(4,49)-(4,52)
(4,53)-(4,64)
(4,54)-(4,55)
(4,58)-(4,63)
(4,59)-(4,60)
(4,61)-(4,62)
(4,68)-(4,69)
*)
