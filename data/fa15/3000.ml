
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
      match a with
      | [] -> []
      | h::t ->
          if (j + k) > 9
          then 1 :: (((h + j) + k) - 10) :: t
          else 0 :: ((h + j) + k) :: t in
    let base = [] in
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
(15,6)-(20,38)
(16,14)-(16,16)
(18,14)-(18,15)
(18,23)-(18,24)
(19,15)-(19,45)
(19,23)-(19,24)
(19,37)-(19,39)
(19,44)-(19,45)
(20,15)-(20,16)
(20,15)-(20,38)
(20,22)-(20,23)
(20,37)-(20,38)
(21,4)-(23,51)
(21,15)-(21,17)
*)

(* type error slice
(13,4)-(23,51)
(13,10)-(20,38)
(15,6)-(20,38)
(15,12)-(15,13)
(23,4)-(23,51)
(23,18)-(23,32)
(23,18)-(23,44)
(23,33)-(23,34)
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
(11,11)-(24,34)
(11,14)-(24,34)
(12,2)-(24,34)
(12,11)-(23,51)
(13,4)-(23,51)
(13,10)-(20,38)
(13,12)-(20,38)
(14,6)-(20,38)
(14,18)-(14,19)
(15,6)-(20,38)
(15,12)-(15,13)
(16,14)-(16,16)
(18,10)-(20,38)
(18,13)-(18,24)
(18,13)-(18,20)
(18,14)-(18,15)
(18,18)-(18,19)
(18,23)-(18,24)
(19,15)-(19,45)
(19,15)-(19,16)
(19,20)-(19,45)
(19,20)-(19,40)
(19,21)-(19,34)
(19,22)-(19,29)
(19,23)-(19,24)
(19,27)-(19,28)
(19,32)-(19,33)
(19,37)-(19,39)
(19,44)-(19,45)
(20,15)-(20,38)
(20,15)-(20,16)
(20,20)-(20,38)
(20,20)-(20,33)
(20,21)-(20,28)
(20,22)-(20,23)
(20,26)-(20,27)
(20,31)-(20,32)
(20,37)-(20,38)
(21,4)-(23,51)
(21,15)-(21,17)
(22,4)-(23,51)
(22,15)-(22,55)
(22,15)-(22,27)
(22,28)-(22,41)
(22,29)-(22,37)
(22,38)-(22,40)
(22,42)-(22,55)
(22,43)-(22,51)
(22,52)-(22,54)
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
