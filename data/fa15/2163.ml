
let rec wwhile (f,b) =
  let (b',c') = f b in if c' = true then wwhile (f, b') else b';;

let fixpoint (f,b) = ((wwhile (fun x  -> ((wwhile (f, b)), (b = (f b))))), b);;


(* fix

let rec wwhile (f,b) =
  let (b',c') = f b in if c' = true then wwhile (f, b') else b';;

let fixpoint (f,b) = wwhile ((fun x  -> ((f x), (b = (f b)))), b);;

*)

(* changed spans
(5,21)-(5,77)
(5,30)-(5,72)
(5,43)-(5,49)
(5,50)-(5,56)
(5,54)-(5,55)
*)

(* type error slice
(3,2)-(3,63)
(3,16)-(3,17)
(3,16)-(3,19)
(3,41)-(3,47)
(3,41)-(3,55)
(3,48)-(3,55)
(3,49)-(3,50)
(3,52)-(3,54)
(5,22)-(5,73)
(5,23)-(5,29)
(5,30)-(5,72)
(5,42)-(5,57)
(5,43)-(5,49)
(5,50)-(5,56)
(5,51)-(5,52)
(5,54)-(5,55)
(5,59)-(5,70)
(5,60)-(5,61)
(5,64)-(5,69)
(5,65)-(5,66)
*)

(* all spans
(2,16)-(3,63)
(3,2)-(3,63)
(3,16)-(3,19)
(3,16)-(3,17)
(3,18)-(3,19)
(3,23)-(3,63)
(3,26)-(3,35)
(3,26)-(3,28)
(3,31)-(3,35)
(3,41)-(3,55)
(3,41)-(3,47)
(3,48)-(3,55)
(3,49)-(3,50)
(3,52)-(3,54)
(3,61)-(3,63)
(5,14)-(5,77)
(5,21)-(5,77)
(5,22)-(5,73)
(5,23)-(5,29)
(5,30)-(5,72)
(5,41)-(5,71)
(5,42)-(5,57)
(5,43)-(5,49)
(5,50)-(5,56)
(5,51)-(5,52)
(5,54)-(5,55)
(5,59)-(5,70)
(5,60)-(5,61)
(5,64)-(5,69)
(5,65)-(5,66)
(5,67)-(5,68)
(5,75)-(5,76)
*)