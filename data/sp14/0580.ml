
let rec wwhile (f,b) =
  let (number,boolean) = f b in
  if boolean then wwhile (f, number) else number;;

let fixpoint (f,b) = wwhile ((fun y  -> fun x  -> ((f x), ((f x) != b))), b);;


(* fix

let rec wwhile (f,b) =
  let (number,boolean) = f b in
  if boolean then wwhile (f, number) else number;;

let fixpoint (f,b) =
  wwhile (let y x = let xx = f x in (xx, (xx != x)) in (y, b));;

*)

(* changed spans
(6,28)-(6,76)
(6,29)-(6,72)
(6,50)-(6,71)
(6,58)-(6,70)
(6,59)-(6,64)
(6,60)-(6,61)
(6,68)-(6,69)
(6,74)-(6,75)
*)

(* type error slice
(3,2)-(4,48)
(3,25)-(3,26)
(3,25)-(3,28)
(4,18)-(4,24)
(4,18)-(4,36)
(4,25)-(4,36)
(4,26)-(4,27)
(6,21)-(6,27)
(6,21)-(6,76)
(6,28)-(6,76)
(6,29)-(6,72)
(6,40)-(6,71)
*)

(* all spans
(2,16)-(4,48)
(3,2)-(4,48)
(3,25)-(3,28)
(3,25)-(3,26)
(3,27)-(3,28)
(4,2)-(4,48)
(4,5)-(4,12)
(4,18)-(4,36)
(4,18)-(4,24)
(4,25)-(4,36)
(4,26)-(4,27)
(4,29)-(4,35)
(4,42)-(4,48)
(6,14)-(6,76)
(6,21)-(6,76)
(6,21)-(6,27)
(6,28)-(6,76)
(6,29)-(6,72)
(6,40)-(6,71)
(6,50)-(6,71)
(6,51)-(6,56)
(6,52)-(6,53)
(6,54)-(6,55)
(6,58)-(6,70)
(6,59)-(6,64)
(6,60)-(6,61)
(6,62)-(6,63)
(6,68)-(6,69)
(6,74)-(6,75)
*)