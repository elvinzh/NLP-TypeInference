
let l1 = [0; 0; 9; 9];;

let l2 = [1; 0; 0; 2];;

let x = (3, 3) :: (List.rev (List.combine l1 l2));;

let clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | x::xs -> if x = 0 then removeZero xs else x :: xs;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = match x with | (c,d) -> (c + d) :: a in
    let base = (0, []) in
    let args = match l1 with | h::t -> [(h, l2)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let l1 = [0; 0; 9; 9];;

let l2 = [1; 0; 0; 2];;

let x = (3, 3) :: (List.rev (List.combine l1 l2));;

let clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | x::xs -> if x = 0 then removeZero xs else x :: xs;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = match x with | (c,d::t) -> a in
    let base = (0, []) in
    let args = match l1 with | h::t -> [(h, l2)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(23,16)-(23,52)
(23,40)-(23,47)
(23,40)-(23,52)
(23,41)-(23,42)
(23,45)-(23,46)
*)

(* type error slice
(23,4)-(26,51)
(23,10)-(23,52)
(23,40)-(23,52)
(23,51)-(23,52)
(24,4)-(26,51)
(24,15)-(24,22)
(26,18)-(26,32)
(26,18)-(26,44)
(26,33)-(26,34)
(26,35)-(26,39)
*)

(* all spans
(2,9)-(2,21)
(2,10)-(2,11)
(2,13)-(2,14)
(2,16)-(2,17)
(2,19)-(2,20)
(4,9)-(4,21)
(4,10)-(4,11)
(4,13)-(4,14)
(4,16)-(4,17)
(4,19)-(4,20)
(6,8)-(6,49)
(6,8)-(6,14)
(6,9)-(6,10)
(6,12)-(6,13)
(6,18)-(6,49)
(6,19)-(6,27)
(6,28)-(6,48)
(6,29)-(6,41)
(6,42)-(6,44)
(6,45)-(6,47)
(8,10)-(11,15)
(8,12)-(11,15)
(9,2)-(11,15)
(9,17)-(10,55)
(9,19)-(10,55)
(9,21)-(10,55)
(10,4)-(10,55)
(10,7)-(10,13)
(10,7)-(10,8)
(10,12)-(10,13)
(10,19)-(10,22)
(10,28)-(10,55)
(10,28)-(10,34)
(10,35)-(10,36)
(10,37)-(10,44)
(10,38)-(10,39)
(10,42)-(10,43)
(10,45)-(10,55)
(10,46)-(10,47)
(10,51)-(10,54)
(11,2)-(11,15)
(11,2)-(11,8)
(11,9)-(11,10)
(11,11)-(11,12)
(11,13)-(11,15)
(13,12)-(16,77)
(13,15)-(16,77)
(14,2)-(16,77)
(14,5)-(14,40)
(14,5)-(14,21)
(14,6)-(14,17)
(14,18)-(14,20)
(14,24)-(14,40)
(14,25)-(14,36)
(14,37)-(14,39)
(15,7)-(15,77)
(15,8)-(15,72)
(15,9)-(15,20)
(15,21)-(15,68)
(15,22)-(15,27)
(15,28)-(15,29)
(15,30)-(15,67)
(15,31)-(15,47)
(15,32)-(15,43)
(15,44)-(15,46)
(15,50)-(15,66)
(15,51)-(15,62)
(15,63)-(15,65)
(15,69)-(15,71)
(15,74)-(15,76)
(16,7)-(16,77)
(16,8)-(16,10)
(16,12)-(16,76)
(16,13)-(16,24)
(16,25)-(16,72)
(16,26)-(16,31)
(16,32)-(16,33)
(16,34)-(16,71)
(16,35)-(16,51)
(16,36)-(16,47)
(16,48)-(16,50)
(16,54)-(16,70)
(16,55)-(16,66)
(16,67)-(16,69)
(16,73)-(16,75)
(18,19)-(19,77)
(19,2)-(19,77)
(19,8)-(19,9)
(19,23)-(19,25)
(19,37)-(19,77)
(19,40)-(19,45)
(19,40)-(19,41)
(19,44)-(19,45)
(19,51)-(19,64)
(19,51)-(19,61)
(19,62)-(19,64)
(19,70)-(19,77)
(19,70)-(19,71)
(19,75)-(19,77)
(21,11)-(27,34)
(21,14)-(27,34)
(22,2)-(27,34)
(22,11)-(26,51)
(23,4)-(26,51)
(23,10)-(23,52)
(23,12)-(23,52)
(23,16)-(23,52)
(23,22)-(23,23)
(23,40)-(23,52)
(23,40)-(23,47)
(23,41)-(23,42)
(23,45)-(23,46)
(23,51)-(23,52)
(24,4)-(26,51)
(24,15)-(24,22)
(24,16)-(24,17)
(24,19)-(24,21)
(25,4)-(26,51)
(25,15)-(25,48)
(25,21)-(25,23)
(25,39)-(25,48)
(25,40)-(25,47)
(25,41)-(25,42)
(25,44)-(25,46)
(26,4)-(26,51)
(26,18)-(26,44)
(26,18)-(26,32)
(26,33)-(26,34)
(26,35)-(26,39)
(26,40)-(26,44)
(26,48)-(26,51)
(27,2)-(27,34)
(27,2)-(27,12)
(27,13)-(27,34)
(27,14)-(27,17)
(27,18)-(27,33)
(27,19)-(27,26)
(27,27)-(27,29)
(27,30)-(27,32)
*)
