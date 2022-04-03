
let rec clone x n =
  let rec helper a b acc = if b > 0 then helper a (b - 1) (a :: acc) else acc in
  helper x n [];;

let padZero l1 l2 =
  let l1_len = List.length l1 in
  let l2_len = List.length l2 in
  let l_diff = l1_len - l2_len in
  if l_diff < 0
  then (((clone 0 (l_diff * (-1))) @ l1), l2)
  else (l1, ((clone 0 l_diff) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = let (a,b) = List.hd x in ([a + 1], [b + 2]) in
    let base = ([], []) in
    let args = [(l1, l2)] in
    let (bar,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  let rec helper a b acc = if b > 0 then helper a (b - 1) (a :: acc) else acc in
  helper x n [];;

let padZero l1 l2 =
  let l1_len = List.length l1 in
  let l2_len = List.length l2 in
  let l_diff = l1_len - l2_len in
  if l_diff < 0
  then (((clone 0 (l_diff * (-1))) @ l1), l2)
  else (l1, ((clone 0 l_diff) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = ([x + 1], [x + 2]) in
    let base = ([], []) in
    let args = l1 in let (bar,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(19,16)-(19,59)
(19,28)-(19,35)
(19,28)-(19,37)
(19,36)-(19,37)
(19,43)-(19,44)
(19,52)-(19,53)
(21,15)-(21,25)
(21,16)-(21,24)
(21,21)-(21,23)
*)

(* type error slice
(19,4)-(22,53)
(19,10)-(19,59)
(19,12)-(19,59)
(19,28)-(19,35)
(19,28)-(19,37)
(19,36)-(19,37)
(21,4)-(22,53)
(21,15)-(21,25)
(21,16)-(21,24)
(22,20)-(22,34)
(22,20)-(22,46)
(22,35)-(22,36)
(22,42)-(22,46)
*)

(* all spans
(2,14)-(4,15)
(2,16)-(4,15)
(3,2)-(4,15)
(3,17)-(3,77)
(3,19)-(3,77)
(3,21)-(3,77)
(3,27)-(3,77)
(3,30)-(3,35)
(3,30)-(3,31)
(3,34)-(3,35)
(3,41)-(3,68)
(3,41)-(3,47)
(3,48)-(3,49)
(3,50)-(3,57)
(3,51)-(3,52)
(3,55)-(3,56)
(3,58)-(3,68)
(3,59)-(3,60)
(3,64)-(3,67)
(3,74)-(3,77)
(4,2)-(4,15)
(4,2)-(4,8)
(4,9)-(4,10)
(4,11)-(4,12)
(4,13)-(4,15)
(6,12)-(12,36)
(6,15)-(12,36)
(7,2)-(12,36)
(7,15)-(7,29)
(7,15)-(7,26)
(7,27)-(7,29)
(8,2)-(12,36)
(8,15)-(8,29)
(8,15)-(8,26)
(8,27)-(8,29)
(9,2)-(12,36)
(9,15)-(9,30)
(9,15)-(9,21)
(9,24)-(9,30)
(10,2)-(12,36)
(10,5)-(10,15)
(10,5)-(10,11)
(10,14)-(10,15)
(11,7)-(11,45)
(11,8)-(11,40)
(11,35)-(11,36)
(11,9)-(11,34)
(11,10)-(11,15)
(11,16)-(11,17)
(11,18)-(11,33)
(11,19)-(11,25)
(11,28)-(11,32)
(11,37)-(11,39)
(11,42)-(11,44)
(12,7)-(12,36)
(12,8)-(12,10)
(12,12)-(12,35)
(12,30)-(12,31)
(12,13)-(12,29)
(12,14)-(12,19)
(12,20)-(12,21)
(12,22)-(12,28)
(12,32)-(12,34)
(14,19)-(15,74)
(15,2)-(15,74)
(15,8)-(15,9)
(15,23)-(15,25)
(15,36)-(15,74)
(15,39)-(15,44)
(15,39)-(15,40)
(15,43)-(15,44)
(15,50)-(15,62)
(15,50)-(15,60)
(15,61)-(15,62)
(15,68)-(15,74)
(15,68)-(15,69)
(15,73)-(15,74)
(17,11)-(23,34)
(17,14)-(23,34)
(18,2)-(23,34)
(18,11)-(22,53)
(19,4)-(22,53)
(19,10)-(19,59)
(19,12)-(19,59)
(19,16)-(19,59)
(19,28)-(19,37)
(19,28)-(19,35)
(19,36)-(19,37)
(19,41)-(19,59)
(19,42)-(19,49)
(19,43)-(19,48)
(19,43)-(19,44)
(19,47)-(19,48)
(19,51)-(19,58)
(19,52)-(19,57)
(19,52)-(19,53)
(19,56)-(19,57)
(20,4)-(22,53)
(20,15)-(20,23)
(20,16)-(20,18)
(20,20)-(20,22)
(21,4)-(22,53)
(21,15)-(21,25)
(21,16)-(21,24)
(21,17)-(21,19)
(21,21)-(21,23)
(22,4)-(22,53)
(22,20)-(22,46)
(22,20)-(22,34)
(22,35)-(22,36)
(22,37)-(22,41)
(22,42)-(22,46)
(22,50)-(22,53)
(23,2)-(23,34)
(23,2)-(23,12)
(23,13)-(23,34)
(23,14)-(23,17)
(23,18)-(23,33)
(23,19)-(23,26)
(23,27)-(23,29)
(23,30)-(23,32)
*)