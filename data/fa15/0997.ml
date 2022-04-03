
let helper (f,b) = let x = f b in match x with | b -> false | _ -> true;;

let rec wwhile (f,b) =
  let (x,y) = f b in match y with | false  -> x | true  -> wwhile (f, x);;

let fixpoint (f,b) = wwhile ((helper (f, b)), b);;


(* fix

let helper (f,b) = let f b = let x = f b in (x, (x != b)) in f;;

let rec wwhile (f,b) =
  let (x,y) = f b in match y with | false  -> x | true  -> wwhile (f, x);;

let fixpoint (f,b) = wwhile ((helper (f, b)), b);;

*)

(* changed spans
(2,19)-(2,71)
(2,34)-(2,71)
(2,54)-(2,59)
(2,67)-(2,71)
(4,16)-(5,72)
*)

(* type error slice
(2,3)-(2,73)
(2,12)-(2,71)
(2,19)-(2,71)
(2,34)-(2,71)
(2,54)-(2,59)
(5,14)-(5,15)
(5,14)-(5,17)
(5,59)-(5,65)
(5,59)-(5,72)
(5,66)-(5,72)
(5,67)-(5,68)
(7,21)-(7,27)
(7,21)-(7,48)
(7,28)-(7,48)
(7,29)-(7,44)
(7,30)-(7,36)
*)

(* all spans
(2,12)-(2,71)
(2,19)-(2,71)
(2,27)-(2,30)
(2,27)-(2,28)
(2,29)-(2,30)
(2,34)-(2,71)
(2,40)-(2,41)
(2,54)-(2,59)
(2,67)-(2,71)
(4,16)-(5,72)
(5,2)-(5,72)
(5,14)-(5,17)
(5,14)-(5,15)
(5,16)-(5,17)
(5,21)-(5,72)
(5,27)-(5,28)
(5,46)-(5,47)
(5,59)-(5,72)
(5,59)-(5,65)
(5,66)-(5,72)
(5,67)-(5,68)
(5,70)-(5,71)
(7,14)-(7,48)
(7,21)-(7,48)
(7,21)-(7,27)
(7,28)-(7,48)
(7,29)-(7,44)
(7,30)-(7,36)
(7,37)-(7,43)
(7,38)-(7,39)
(7,41)-(7,42)
(7,46)-(7,47)
*)
