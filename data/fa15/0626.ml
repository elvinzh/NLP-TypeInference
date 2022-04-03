
let rec clone x n = if n > 0 then List.append [x] (clone x (n - 1)) else [];;

let padZero l1 l2 =
  ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1),
    (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with
  | [] -> []
  | _ -> let h::t = l in (match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (a1,a2) = a in
      let (x1,x2) = x in
      let val1 = (a1 + x1) + x2 in
      if val1 > 9 then (1, ((val1 - 10) :: a2)) else (0, (val1 :: a2)) in
    let base = (0, []) in
    let args = List.rev ((List.combine [0]) @ ((l1 [0]) @ l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n > 0 then List.append [x] (clone x (n - 1)) else [];;

let padZero l1 l2 =
  ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1),
    (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with
  | [] -> []
  | _ -> let h::t = l in (match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (a1,a2) = a in
      let (x1,x2) = x in
      let val1 = (a1 + x1) + x2 in
      if val1 > 9 then (1, ((val1 - 10) :: a2)) else (0, (val1 :: a2)) in
    let base = (0, []) in
    let args = List.rev (List.combine (0 :: l1) (0 :: l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(21,25)-(21,43)
(21,39)-(21,42)
(21,44)-(21,45)
(21,46)-(21,61)
(21,47)-(21,55)
(21,51)-(21,54)
(21,56)-(21,57)
*)

(* type error slice
(4,3)-(6,71)
(4,12)-(6,69)
(4,15)-(6,69)
(5,2)-(6,69)
(5,3)-(5,67)
(5,4)-(5,15)
(14,2)-(23,34)
(14,11)-(22,51)
(21,24)-(21,62)
(21,25)-(21,43)
(21,26)-(21,38)
(21,44)-(21,45)
(21,47)-(21,55)
(21,48)-(21,50)
(23,13)-(23,34)
(23,14)-(23,17)
(23,18)-(23,33)
(23,19)-(23,26)
*)

(* all spans
(2,14)-(2,75)
(2,16)-(2,75)
(2,20)-(2,75)
(2,23)-(2,28)
(2,23)-(2,24)
(2,27)-(2,28)
(2,34)-(2,67)
(2,34)-(2,45)
(2,46)-(2,49)
(2,47)-(2,48)
(2,50)-(2,67)
(2,51)-(2,56)
(2,57)-(2,58)
(2,59)-(2,66)
(2,60)-(2,61)
(2,64)-(2,65)
(2,73)-(2,75)
(4,12)-(6,69)
(4,15)-(6,69)
(5,2)-(6,69)
(5,3)-(5,67)
(5,4)-(5,15)
(5,16)-(5,63)
(5,17)-(5,22)
(5,23)-(5,24)
(5,25)-(5,62)
(5,26)-(5,42)
(5,27)-(5,38)
(5,39)-(5,41)
(5,45)-(5,61)
(5,46)-(5,57)
(5,58)-(5,60)
(5,64)-(5,66)
(6,4)-(6,68)
(6,5)-(6,16)
(6,17)-(6,64)
(6,18)-(6,23)
(6,24)-(6,25)
(6,26)-(6,63)
(6,27)-(6,43)
(6,28)-(6,39)
(6,40)-(6,42)
(6,46)-(6,62)
(6,47)-(6,58)
(6,59)-(6,61)
(6,65)-(6,67)
(8,19)-(11,68)
(9,2)-(11,68)
(9,8)-(9,9)
(10,10)-(10,12)
(11,9)-(11,68)
(11,20)-(11,21)
(11,25)-(11,68)
(11,32)-(11,33)
(11,46)-(11,58)
(11,46)-(11,56)
(11,57)-(11,58)
(11,66)-(11,67)
(13,11)-(23,34)
(13,14)-(23,34)
(14,2)-(23,34)
(14,11)-(22,51)
(15,4)-(22,51)
(15,10)-(19,70)
(15,12)-(19,70)
(16,6)-(19,70)
(16,20)-(16,21)
(17,6)-(19,70)
(17,20)-(17,21)
(18,6)-(19,70)
(18,17)-(18,31)
(18,17)-(18,26)
(18,18)-(18,20)
(18,23)-(18,25)
(18,29)-(18,31)
(19,6)-(19,70)
(19,9)-(19,17)
(19,9)-(19,13)
(19,16)-(19,17)
(19,23)-(19,47)
(19,24)-(19,25)
(19,27)-(19,46)
(19,28)-(19,39)
(19,29)-(19,33)
(19,36)-(19,38)
(19,43)-(19,45)
(19,53)-(19,70)
(19,54)-(19,55)
(19,57)-(19,69)
(19,58)-(19,62)
(19,66)-(19,68)
(20,4)-(22,51)
(20,15)-(20,22)
(20,16)-(20,17)
(20,19)-(20,21)
(21,4)-(22,51)
(21,15)-(21,62)
(21,15)-(21,23)
(21,24)-(21,62)
(21,44)-(21,45)
(21,25)-(21,43)
(21,26)-(21,38)
(21,39)-(21,42)
(21,40)-(21,41)
(21,46)-(21,61)
(21,56)-(21,57)
(21,47)-(21,55)
(21,48)-(21,50)
(21,51)-(21,54)
(21,52)-(21,53)
(21,58)-(21,60)
(22,4)-(22,51)
(22,18)-(22,44)
(22,18)-(22,32)
(22,33)-(22,34)
(22,35)-(22,39)
(22,40)-(22,44)
(22,48)-(22,51)
(23,2)-(23,34)
(23,2)-(23,12)
(23,13)-(23,34)
(23,14)-(23,17)
(23,18)-(23,33)
(23,19)-(23,26)
(23,27)-(23,29)
(23,30)-(23,32)
*)
