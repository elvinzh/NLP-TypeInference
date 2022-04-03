
let rec mulByDigit i l =
  let f a xs =
    let (a1,a2) = a in
    let h::t = xs in
    let val1 = (h * i) + a1 in
    if val1 > 9
    then ((val1 / 10), ((val1 mod 10) :: a2))
    else (0, (val1 :: a2)) in
  let base = (0, []) in
  let args = 0 :: (List.rev l) in
  let (_,res) = List.fold_left f base args in res;;


(* fix

let rec mulByDigit i l =
  let f a x =
    let (a1,a2) = a in
    let val1 = (x * i) + a1 in
    if val1 > 9
    then ((val1 / 10), ((val1 mod 10) :: a2))
    else (0, (val1 :: a2)) in
  let base = (0, []) in
  let args = 0 :: (List.rev l) in
  let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(3,10)-(9,26)
(5,4)-(9,26)
(5,15)-(5,17)
(6,16)-(6,17)
*)

(* type error slice
(3,2)-(12,49)
(3,8)-(9,26)
(3,10)-(9,26)
(5,4)-(9,26)
(5,15)-(5,17)
(11,2)-(12,49)
(11,13)-(11,14)
(11,13)-(11,30)
(12,16)-(12,30)
(12,16)-(12,42)
(12,31)-(12,32)
(12,38)-(12,42)
*)

(* all spans
(2,19)-(12,49)
(2,21)-(12,49)
(3,2)-(12,49)
(3,8)-(9,26)
(3,10)-(9,26)
(4,4)-(9,26)
(4,18)-(4,19)
(5,4)-(9,26)
(5,15)-(5,17)
(6,4)-(9,26)
(6,15)-(6,27)
(6,15)-(6,22)
(6,16)-(6,17)
(6,20)-(6,21)
(6,25)-(6,27)
(7,4)-(9,26)
(7,7)-(7,15)
(7,7)-(7,11)
(7,14)-(7,15)
(8,9)-(8,45)
(8,10)-(8,21)
(8,11)-(8,15)
(8,18)-(8,20)
(8,23)-(8,44)
(8,24)-(8,37)
(8,25)-(8,29)
(8,34)-(8,36)
(8,41)-(8,43)
(9,9)-(9,26)
(9,10)-(9,11)
(9,13)-(9,25)
(9,14)-(9,18)
(9,22)-(9,24)
(10,2)-(12,49)
(10,13)-(10,20)
(10,14)-(10,15)
(10,17)-(10,19)
(11,2)-(12,49)
(11,13)-(11,30)
(11,13)-(11,14)
(11,18)-(11,30)
(11,19)-(11,27)
(11,28)-(11,29)
(12,2)-(12,49)
(12,16)-(12,42)
(12,16)-(12,30)
(12,31)-(12,32)
(12,33)-(12,37)
(12,38)-(12,42)
(12,46)-(12,49)
*)
