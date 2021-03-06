
let rec wwhile (f,b) = let (b',c') = f b in if c' then wwhile (f, b') else b';;

let fixpoint (f,b) = let b' = (b, ((f b) < b)) in wwhile (f, b');;


(* fix

let rec wwhile (f,b) = let (b',c') = f b in if c' then wwhile (f, b') else b';;

let fixpoint (f,b) = let f b = ((f b), (b = (f b))) in wwhile (f, b);;

*)

(* changed spans
(4,21)-(4,64)
(4,30)-(4,46)
(4,31)-(4,32)
(4,34)-(4,45)
(4,43)-(4,44)
(4,50)-(4,56)
(4,61)-(4,63)
*)

(* type error slice
(2,23)-(2,77)
(2,37)-(2,38)
(2,37)-(2,40)
(2,55)-(2,61)
(2,55)-(2,69)
(2,62)-(2,69)
(2,63)-(2,64)
(2,66)-(2,68)
(4,21)-(4,64)
(4,30)-(4,46)
(4,31)-(4,32)
(4,34)-(4,45)
(4,35)-(4,40)
(4,36)-(4,37)
(4,43)-(4,44)
(4,50)-(4,56)
(4,50)-(4,64)
(4,57)-(4,64)
(4,58)-(4,59)
(4,61)-(4,63)
*)

(* all spans
(2,16)-(2,77)
(2,23)-(2,77)
(2,37)-(2,40)
(2,37)-(2,38)
(2,39)-(2,40)
(2,44)-(2,77)
(2,47)-(2,49)
(2,55)-(2,69)
(2,55)-(2,61)
(2,62)-(2,69)
(2,63)-(2,64)
(2,66)-(2,68)
(2,75)-(2,77)
(4,14)-(4,64)
(4,21)-(4,64)
(4,30)-(4,46)
(4,31)-(4,32)
(4,34)-(4,45)
(4,35)-(4,40)
(4,36)-(4,37)
(4,38)-(4,39)
(4,43)-(4,44)
(4,50)-(4,64)
(4,50)-(4,56)
(4,57)-(4,64)
(4,58)-(4,59)
(4,61)-(4,63)
*)
