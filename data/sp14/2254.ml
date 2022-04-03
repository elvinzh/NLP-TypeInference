
let rec wwhile (f,b) =
  let (b',c') = f b in match c' with | false  -> b' | _ -> wwhile (f, b');;

let fixpoint (f,b) =
  wwhile
    ((fun (f,b)  -> match f b with | b -> (b, false) | _ -> ((f b), true)),
      b);;


(* fix

let rec wwhile (f,b) =
  let (b',c') = f b in match c' with | false  -> b' | _ -> wwhile (f, b');;

let fixpoint (f,b) =
  wwhile ((fun a  -> if b = (f b) then (b, false) else ((f b), true)), b);;

*)

(* changed spans
(7,5)-(7,74)
(7,20)-(7,73)
(7,26)-(7,29)
*)

(* type error slice
(2,3)-(3,75)
(2,16)-(3,73)
(3,2)-(3,73)
(3,16)-(3,17)
(3,16)-(3,19)
(3,18)-(3,19)
(3,59)-(3,65)
(3,59)-(3,73)
(3,66)-(3,73)
(3,70)-(3,72)
(6,2)-(6,8)
(6,2)-(8,8)
(7,4)-(8,8)
(7,5)-(7,74)
(7,20)-(7,73)
(7,26)-(7,27)
(7,26)-(7,29)
(7,42)-(7,52)
(7,43)-(7,44)
*)

(* all spans
(2,16)-(3,73)
(3,2)-(3,73)
(3,16)-(3,19)
(3,16)-(3,17)
(3,18)-(3,19)
(3,23)-(3,73)
(3,29)-(3,31)
(3,49)-(3,51)
(3,59)-(3,73)
(3,59)-(3,65)
(3,66)-(3,73)
(3,67)-(3,68)
(3,70)-(3,72)
(5,14)-(8,8)
(6,2)-(8,8)
(6,2)-(6,8)
(7,4)-(8,8)
(7,5)-(7,74)
(7,20)-(7,73)
(7,26)-(7,29)
(7,26)-(7,27)
(7,28)-(7,29)
(7,42)-(7,52)
(7,43)-(7,44)
(7,46)-(7,51)
(7,60)-(7,73)
(7,61)-(7,66)
(7,62)-(7,63)
(7,64)-(7,65)
(7,68)-(7,72)
(8,6)-(8,7)
*)
