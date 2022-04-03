
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
    let f a x = failwith "to be implemented" in
    let base = [] in
    let args =
      let combine (a,b) = a + b in
      List.map combine (List.rev (List.combine l1 l2)) in
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
(18,16)-(18,24)
(18,16)-(18,44)
(18,25)-(18,44)
(19,4)-(23,51)
(19,15)-(19,17)
*)

(* type error slice
(19,4)-(23,51)
(19,15)-(19,17)
(23,4)-(23,51)
(23,18)-(23,32)
(23,18)-(23,44)
(23,35)-(23,39)
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
(16,11)-(24,34)
(16,14)-(24,34)
(17,2)-(24,34)
(17,11)-(23,51)
(18,4)-(23,51)
(18,10)-(18,44)
(18,12)-(18,44)
(18,16)-(18,44)
(18,16)-(18,24)
(18,25)-(18,44)
(19,4)-(23,51)
(19,15)-(19,17)
(20,4)-(23,51)
(21,6)-(22,54)
(21,19)-(21,31)
(21,26)-(21,31)
(21,26)-(21,27)
(21,30)-(21,31)
(22,6)-(22,54)
(22,6)-(22,14)
(22,15)-(22,22)
(22,23)-(22,54)
(22,24)-(22,32)
(22,33)-(22,53)
(22,34)-(22,46)
(22,47)-(22,49)
(22,50)-(22,52)
(23,4)-(23,51)
(23,18)-(23,44)
(23,18)-(23,32)
(23,33)-(23,34)
(23,35)-(23,39)
(23,40)-(23,44)
(23,48)-(23,51)
(24,2)-(24,34)
(24,2)-(24,12)
(24,13)-(24,34)
(24,14)-(24,17)
(24,18)-(24,33)
(24,19)-(24,26)
(24,27)-(24,29)
(24,30)-(24,32)
*)
