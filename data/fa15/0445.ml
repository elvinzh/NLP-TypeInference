
let rec wwhile (f,b) =
  let res = f b in
  match res with | (x,y) when y = true -> wwhile (f, x) | (x,y) -> x;;

let fixpoint (f,b) =
  let gs x =
    let xx = f x in match xx with | xx when (xx - x) > 0 -> (x, b) | _ -> f x in
  wwhile (gs, b);;


(* fix

let rec wwhile (f,b) =
  let res = f b in
  match res with | (x,y) when y = true -> wwhile (f, x) | (x,y) -> x;;

let fixpoint (f,b) =
  let gs x = let xx = f x in (xx, (xx = x)) in wwhile (gs, b);;

*)

(* changed spans
(8,20)-(8,77)
(8,26)-(8,28)
(8,45)-(8,47)
(8,61)-(8,62)
(8,64)-(8,65)
(8,74)-(8,75)
(8,74)-(8,77)
*)

(* type error slice
(3,2)-(4,68)
(3,12)-(3,13)
(3,12)-(3,15)
(4,2)-(4,68)
(4,8)-(4,11)
(4,30)-(4,31)
(4,30)-(4,38)
(4,34)-(4,38)
(4,42)-(4,48)
(4,42)-(4,55)
(4,49)-(4,55)
(4,50)-(4,51)
(4,53)-(4,54)
(7,2)-(9,16)
(7,9)-(8,77)
(8,4)-(8,77)
(8,13)-(8,14)
(8,13)-(8,16)
(8,20)-(8,77)
(8,26)-(8,28)
(8,44)-(8,52)
(8,45)-(8,47)
(8,50)-(8,51)
(8,60)-(8,66)
(8,61)-(8,62)
(8,64)-(8,65)
(8,74)-(8,75)
(8,74)-(8,77)
(9,2)-(9,8)
(9,2)-(9,16)
(9,9)-(9,16)
(9,10)-(9,12)
(9,14)-(9,15)
*)

(* all spans
(2,16)-(4,68)
(3,2)-(4,68)
(3,12)-(3,15)
(3,12)-(3,13)
(3,14)-(3,15)
(4,2)-(4,68)
(4,8)-(4,11)
(4,30)-(4,38)
(4,30)-(4,31)
(4,34)-(4,38)
(4,42)-(4,55)
(4,42)-(4,48)
(4,49)-(4,55)
(4,50)-(4,51)
(4,53)-(4,54)
(4,67)-(4,68)
(6,14)-(9,16)
(7,2)-(9,16)
(7,9)-(8,77)
(8,4)-(8,77)
(8,13)-(8,16)
(8,13)-(8,14)
(8,15)-(8,16)
(8,20)-(8,77)
(8,26)-(8,28)
(8,44)-(8,56)
(8,44)-(8,52)
(8,45)-(8,47)
(8,50)-(8,51)
(8,55)-(8,56)
(8,60)-(8,66)
(8,61)-(8,62)
(8,64)-(8,65)
(8,74)-(8,77)
(8,74)-(8,75)
(8,76)-(8,77)
(9,2)-(9,16)
(9,2)-(9,8)
(9,9)-(9,16)
(9,10)-(9,12)
(9,14)-(9,15)
*)
