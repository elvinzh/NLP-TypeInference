
let bigMul l1 l2 =
  let f a x =
    let (m,n) = a in
    let (c,d) = x in let z = (c * d) + m in ((z / 10), ((z mod 10) :: n)) in
  let base = (0, []) in
  let args =
    List.combine (List.rev ([(0, 0, 0, 0)] :: l1))
      (List.rev ([(0, 0, 0, 0)] :: l2)) in
  let (_,res) = List.fold_left f base args in res;;


(* fix

let bigMul l1 l2 =
  let f a x =
    let (m,n) = a in
    let (c,d) = x in let z = (c * d) + m in ((z / 10), ((z mod 10) :: n)) in
  let base = (0, []) in
  let args =
    List.combine (List.rev (0 :: 0 :: 0 :: 0 :: l1))
      (List.rev (0 :: 0 :: 0 :: 0 :: l2)) in
  let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(8,28)-(8,42)
(8,29)-(8,41)
(8,33)-(8,34)
(8,36)-(8,37)
(8,39)-(8,40)
(9,17)-(9,31)
(9,18)-(9,30)
(9,22)-(9,23)
(9,25)-(9,26)
(9,28)-(9,29)
*)

(* type error slice
(3,2)-(10,49)
(3,8)-(5,73)
(3,10)-(5,73)
(5,4)-(5,73)
(5,16)-(5,17)
(5,29)-(5,36)
(5,34)-(5,35)
(7,2)-(10,49)
(8,4)-(8,16)
(8,4)-(9,39)
(9,6)-(9,39)
(9,7)-(9,15)
(9,16)-(9,38)
(9,17)-(9,31)
(10,16)-(10,30)
(10,16)-(10,42)
(10,31)-(10,32)
(10,38)-(10,42)
*)

(* all spans
(2,11)-(10,49)
(2,14)-(10,49)
(3,2)-(10,49)
(3,8)-(5,73)
(3,10)-(5,73)
(4,4)-(5,73)
(4,16)-(4,17)
(5,4)-(5,73)
(5,16)-(5,17)
(5,21)-(5,73)
(5,29)-(5,40)
(5,29)-(5,36)
(5,30)-(5,31)
(5,34)-(5,35)
(5,39)-(5,40)
(5,44)-(5,73)
(5,45)-(5,53)
(5,46)-(5,47)
(5,50)-(5,52)
(5,55)-(5,72)
(5,56)-(5,66)
(5,57)-(5,58)
(5,63)-(5,65)
(5,70)-(5,71)
(6,2)-(10,49)
(6,13)-(6,20)
(6,14)-(6,15)
(6,17)-(6,19)
(7,2)-(10,49)
(8,4)-(9,39)
(8,4)-(8,16)
(8,17)-(8,50)
(8,18)-(8,26)
(8,27)-(8,49)
(8,28)-(8,42)
(8,29)-(8,41)
(8,30)-(8,31)
(8,33)-(8,34)
(8,36)-(8,37)
(8,39)-(8,40)
(8,46)-(8,48)
(9,6)-(9,39)
(9,7)-(9,15)
(9,16)-(9,38)
(9,17)-(9,31)
(9,18)-(9,30)
(9,19)-(9,20)
(9,22)-(9,23)
(9,25)-(9,26)
(9,28)-(9,29)
(9,35)-(9,37)
(10,2)-(10,49)
(10,16)-(10,42)
(10,16)-(10,30)
(10,31)-(10,32)
(10,33)-(10,37)
(10,38)-(10,42)
(10,46)-(10,49)
*)
