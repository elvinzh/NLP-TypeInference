
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let difference = (List.length l1) - (List.length l2) in
  if difference > 0
  then (l1, ((clone 0 difference) @ l2))
  else
    if difference < 0
    then (((clone 0 ((-1) * difference)) @ l1), l2)
    else (l1, l2);;

let rec removeZero l =
  match l with | [] -> l | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let sum = match x with | (x1,x2) -> x1 + x2 in if sum < 10 then x :: a in
    let base = [] in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let difference = (List.length l1) - (List.length l2) in
  if difference > 0
  then (l1, ((clone 0 difference) @ l2))
  else
    if difference < 0
    then (((clone 0 ((-1) * difference)) @ l1), l2)
    else (l1, l2);;

let rec removeZero l =
  match l with | [] -> l | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = match a with | (o,[]) -> (o, [x]) | (o,l) -> (o, (x :: l)) in
    let base = (0, []) in
    let args =
      let combine (a,b) = a + b in
      List.map combine (List.rev (List.combine l1 l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(19,6)-(19,76)
(19,16)-(19,49)
(19,22)-(19,23)
(19,42)-(19,44)
(19,42)-(19,49)
(19,47)-(19,49)
(19,53)-(19,76)
(19,56)-(19,59)
(19,56)-(19,64)
(19,62)-(19,64)
(19,70)-(19,76)
(19,75)-(19,76)
(20,15)-(20,17)
(21,15)-(21,44)
*)

(* type error slice
(19,53)-(19,76)
(19,70)-(19,76)
*)

(* all spans
(2,14)-(2,65)
(2,16)-(2,65)
(2,20)-(2,65)
(2,23)-(2,29)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,37)
(2,43)-(2,65)
(2,43)-(2,44)
(2,48)-(2,65)
(2,49)-(2,54)
(2,55)-(2,56)
(2,57)-(2,64)
(2,58)-(2,59)
(2,62)-(2,63)
(4,12)-(11,17)
(4,15)-(11,17)
(5,2)-(11,17)
(5,19)-(5,54)
(5,19)-(5,35)
(5,20)-(5,31)
(5,32)-(5,34)
(5,38)-(5,54)
(5,39)-(5,50)
(5,51)-(5,53)
(6,2)-(11,17)
(6,5)-(6,19)
(6,5)-(6,15)
(6,18)-(6,19)
(7,7)-(7,40)
(7,8)-(7,10)
(7,12)-(7,39)
(7,34)-(7,35)
(7,13)-(7,33)
(7,14)-(7,19)
(7,20)-(7,21)
(7,22)-(7,32)
(7,36)-(7,38)
(9,4)-(11,17)
(9,7)-(9,21)
(9,7)-(9,17)
(9,20)-(9,21)
(10,9)-(10,51)
(10,10)-(10,46)
(10,41)-(10,42)
(10,11)-(10,40)
(10,12)-(10,17)
(10,18)-(10,19)
(10,20)-(10,39)
(10,21)-(10,25)
(10,28)-(10,38)
(10,43)-(10,45)
(10,48)-(10,50)
(11,9)-(11,17)
(11,10)-(11,12)
(11,14)-(11,16)
(13,19)-(14,73)
(14,2)-(14,73)
(14,8)-(14,9)
(14,23)-(14,24)
(14,35)-(14,73)
(14,38)-(14,43)
(14,38)-(14,39)
(14,42)-(14,43)
(14,49)-(14,61)
(14,49)-(14,59)
(14,60)-(14,61)
(14,67)-(14,73)
(14,67)-(14,68)
(14,72)-(14,73)
(16,11)-(23,34)
(16,14)-(23,34)
(17,2)-(23,34)
(17,11)-(22,51)
(18,4)-(22,51)
(18,10)-(19,76)
(18,12)-(19,76)
(19,6)-(19,76)
(19,16)-(19,49)
(19,22)-(19,23)
(19,42)-(19,49)
(19,42)-(19,44)
(19,47)-(19,49)
(19,53)-(19,76)
(19,56)-(19,64)
(19,56)-(19,59)
(19,62)-(19,64)
(19,70)-(19,76)
(19,70)-(19,71)
(19,75)-(19,76)
(20,4)-(22,51)
(20,15)-(20,17)
(21,4)-(22,51)
(21,15)-(21,44)
(21,15)-(21,23)
(21,24)-(21,44)
(21,25)-(21,37)
(21,38)-(21,40)
(21,41)-(21,43)
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
