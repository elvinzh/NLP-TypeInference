
let rec wwhile (f,b) =
  let (number,boolean) = f b in
  if boolean then wwhile (f, number) else number;;

let fixpoint (f,b) =
  wwhile
    (let f x = let xx = (x * x) * x in (xx, (xx < 100)) in
     ((wwhile (f, 2)), b));;


(* fix

let rec wwhile (f,b) =
  let (number,boolean) = f b in
  if boolean then wwhile (f, number) else number;;

let fixpoint (f,b) =
  wwhile (let y x = let xx = f x in (xx, (xx != x)) in (y, b));;

*)

(* changed spans
(8,4)-(9,26)
(8,24)-(8,31)
(8,24)-(8,35)
(8,25)-(8,26)
(8,29)-(8,30)
(8,34)-(8,35)
(8,44)-(8,54)
(8,50)-(8,53)
(9,5)-(9,25)
(9,6)-(9,21)
(9,7)-(9,13)
(9,15)-(9,16)
(9,18)-(9,19)
*)

(* type error slice
(4,2)-(4,48)
(4,18)-(4,24)
(4,18)-(4,36)
(4,25)-(4,36)
(4,29)-(4,35)
(4,42)-(4,48)
(7,2)-(7,8)
(7,2)-(9,26)
(8,4)-(9,26)
(8,11)-(8,55)
(9,5)-(9,25)
(9,6)-(9,21)
(9,7)-(9,13)
(9,14)-(9,20)
(9,15)-(9,16)
(9,18)-(9,19)
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
(6,14)-(9,26)
(7,2)-(9,26)
(7,2)-(7,8)
(8,4)-(9,26)
(8,11)-(8,55)
(8,15)-(8,55)
(8,24)-(8,35)
(8,24)-(8,31)
(8,25)-(8,26)
(8,29)-(8,30)
(8,34)-(8,35)
(8,39)-(8,55)
(8,40)-(8,42)
(8,44)-(8,54)
(8,45)-(8,47)
(8,50)-(8,53)
(9,5)-(9,25)
(9,6)-(9,21)
(9,7)-(9,13)
(9,14)-(9,20)
(9,15)-(9,16)
(9,18)-(9,19)
(9,23)-(9,24)
*)