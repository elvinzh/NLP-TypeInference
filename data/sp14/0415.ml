
let rec wwhile (f,b) = let (b',c') = f b in if c' then wwhile (f, b') else b';;

let fixpoint (f,b) =
  wwhile ((let w b' = let fb = f b' in (fb, (fb = b')) in f b), b);;


(* fix

let rec wwhile (f,b) = let (b',c') = f b in if c' then wwhile (f, b') else b';;

let fixpoint (f,b) =
  let w b' = let fb = f b' in (fb, (fb = b')) in wwhile (w, b);;

*)

(* changed spans
(5,2)-(5,8)
(5,2)-(5,66)
(5,9)-(5,66)
(5,58)-(5,59)
(5,60)-(5,61)
(5,64)-(5,65)
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
(5,2)-(5,8)
(5,2)-(5,66)
(5,9)-(5,66)
(5,10)-(5,62)
(5,22)-(5,54)
(5,31)-(5,32)
(5,31)-(5,35)
(5,33)-(5,35)
(5,44)-(5,53)
(5,45)-(5,47)
(5,50)-(5,52)
(5,58)-(5,59)
(5,58)-(5,61)
(5,60)-(5,61)
(5,64)-(5,65)
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
(4,14)-(5,66)
(5,2)-(5,66)
(5,2)-(5,8)
(5,9)-(5,66)
(5,10)-(5,62)
(5,17)-(5,54)
(5,22)-(5,54)
(5,31)-(5,35)
(5,31)-(5,32)
(5,33)-(5,35)
(5,39)-(5,54)
(5,40)-(5,42)
(5,44)-(5,53)
(5,45)-(5,47)
(5,50)-(5,52)
(5,58)-(5,61)
(5,58)-(5,59)
(5,60)-(5,61)
(5,64)-(5,65)
*)