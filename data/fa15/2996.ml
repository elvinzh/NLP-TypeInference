
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let d = (List.length l1) - (List.length l2) in
  if d < 0 then (((clone 0 (0 - d)) @ l1), l2) else (l1, ((clone 0 d) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | [] -> []
      | h::t ->
          (match x with
           | (j,k) ->
               if (j + k) > 9
               then 1 :: (((h + j) + k) - 10) :: t
               else 0 :: ((h + j) + k) :: t) in
    let base = [0] in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let d = (List.length l1) - (List.length l2) in
  if d < 0 then (((clone 0 (0 - d)) @ l1), l2) else (l1, ((clone 0 d) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (j,k) = x in
      let (l,m) = a in
      if ((j + k) + l) > 9
      then (1, ((((j + k) + l) - 10) :: m))
      else (0, (((j + k) + l) :: m)) in
    let base = (0, []) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(14,6)-(21,44)
(14,12)-(14,13)
(15,14)-(15,16)
(17,10)-(21,44)
(19,15)-(21,43)
(19,19)-(19,20)
(19,28)-(19,29)
(20,20)-(20,50)
(20,28)-(20,29)
(20,42)-(20,44)
(20,49)-(20,50)
(21,20)-(21,21)
(21,20)-(21,43)
(21,27)-(21,28)
(21,42)-(21,43)
(22,4)-(24,51)
(22,15)-(22,18)
(23,4)-(24,51)
*)

(* type error slice
(13,4)-(24,51)
(13,10)-(21,44)
(14,6)-(21,44)
(14,12)-(14,13)
(24,4)-(24,51)
(24,18)-(24,32)
(24,18)-(24,44)
(24,33)-(24,34)
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
(4,12)-(6,76)
(4,15)-(6,76)
(5,2)-(6,76)
(5,10)-(5,45)
(5,10)-(5,26)
(5,11)-(5,22)
(5,23)-(5,25)
(5,29)-(5,45)
(5,30)-(5,41)
(5,42)-(5,44)
(6,2)-(6,76)
(6,5)-(6,10)
(6,5)-(6,6)
(6,9)-(6,10)
(6,16)-(6,46)
(6,17)-(6,41)
(6,36)-(6,37)
(6,18)-(6,35)
(6,19)-(6,24)
(6,25)-(6,26)
(6,27)-(6,34)
(6,28)-(6,29)
(6,32)-(6,33)
(6,38)-(6,40)
(6,43)-(6,45)
(6,52)-(6,76)
(6,53)-(6,55)
(6,57)-(6,75)
(6,70)-(6,71)
(6,58)-(6,69)
(6,59)-(6,64)
(6,65)-(6,66)
(6,67)-(6,68)
(6,72)-(6,74)
(8,19)-(9,69)
(9,2)-(9,69)
(9,8)-(9,9)
(9,23)-(9,25)
(9,36)-(9,69)
(9,39)-(9,44)
(9,39)-(9,40)
(9,43)-(9,44)
(9,50)-(9,62)
(9,50)-(9,60)
(9,61)-(9,62)
(9,68)-(9,69)
(11,11)-(25,34)
(11,14)-(25,34)
(12,2)-(25,34)
(12,11)-(24,51)
(13,4)-(24,51)
(13,10)-(21,44)
(13,12)-(21,44)
(14,6)-(21,44)
(14,12)-(14,13)
(15,14)-(15,16)
(17,10)-(21,44)
(17,17)-(17,18)
(19,15)-(21,43)
(19,18)-(19,29)
(19,18)-(19,25)
(19,19)-(19,20)
(19,23)-(19,24)
(19,28)-(19,29)
(20,20)-(20,50)
(20,20)-(20,21)
(20,25)-(20,50)
(20,25)-(20,45)
(20,26)-(20,39)
(20,27)-(20,34)
(20,28)-(20,29)
(20,32)-(20,33)
(20,37)-(20,38)
(20,42)-(20,44)
(20,49)-(20,50)
(21,20)-(21,43)
(21,20)-(21,21)
(21,25)-(21,43)
(21,25)-(21,38)
(21,26)-(21,33)
(21,27)-(21,28)
(21,31)-(21,32)
(21,36)-(21,37)
(21,42)-(21,43)
(22,4)-(24,51)
(22,15)-(22,18)
(22,16)-(22,17)
(23,4)-(24,51)
(23,15)-(23,55)
(23,15)-(23,27)
(23,28)-(23,41)
(23,29)-(23,37)
(23,38)-(23,40)
(23,42)-(23,55)
(23,43)-(23,51)
(23,52)-(23,54)
(24,4)-(24,51)
(24,18)-(24,44)
(24,18)-(24,32)
(24,33)-(24,34)
(24,35)-(24,39)
(24,40)-(24,44)
(24,48)-(24,51)
(25,2)-(25,34)
(25,2)-(25,12)
(25,13)-(25,34)
(25,14)-(25,17)
(25,18)-(25,33)
(25,19)-(25,26)
(25,27)-(25,29)
(25,30)-(25,32)
*)
