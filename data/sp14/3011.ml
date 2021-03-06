
let rec clone x n =
  let rec clone_RT acc n =
    if n <= 0 then acc else clone_RT (x :: acc) (n - 1) in
  clone_RT [] n;;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  let diff = len1 - len2 in
  if diff < 0
  then ((List.append (clone 0 (- diff)) l1), l2)
  else (l1, (List.append (clone 0 diff) l2));;

let rec removeZero l =
  match l with | [] -> [] | x::xs -> if x = 0 then removeZero xs else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (([],_),y) -> (([], 0), y)
      | ((h::t,carry),y) ->
          let sum = (h + x) + carry in ((t, (sum / 10)), ((sum mod 10) :: y)) in
    let base = ((0 :: ((List.rev l1), 0)), []) in
    let args = 0 :: (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  let rec clone_RT acc n =
    if n <= 0 then acc else clone_RT (x :: acc) (n - 1) in
  clone_RT [] n;;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  let diff = len1 - len2 in
  if diff < 0
  then ((List.append (clone 0 (- diff)) l1), l2)
  else (l1, (List.append (clone 0 diff) l2));;

let rec removeZero l =
  match l with | [] -> [] | x::xs -> if x = 0 then removeZero xs else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (([],_),y) -> (([], 0), y)
      | ((h::t,carry),y) ->
          let sum = (h + x) + carry in ((t, (sum / 10)), ((sum mod 10) :: y)) in
    let base = (((0 :: (List.rev l1)), 0), []) in
    let args = 0 :: (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(25,16)-(25,41)
(25,22)-(25,40)
*)

(* type error slice
(20,4)-(27,51)
(20,10)-(24,77)
(21,6)-(24,77)
(21,12)-(21,13)
(25,4)-(27,51)
(25,15)-(25,46)
(25,16)-(25,41)
(25,22)-(25,40)
(27,18)-(27,32)
(27,18)-(27,44)
(27,33)-(27,34)
(27,35)-(27,39)
*)

(* all spans
(2,14)-(5,15)
(2,16)-(5,15)
(3,2)-(5,15)
(3,19)-(4,55)
(3,23)-(4,55)
(4,4)-(4,55)
(4,7)-(4,13)
(4,7)-(4,8)
(4,12)-(4,13)
(4,19)-(4,22)
(4,28)-(4,55)
(4,28)-(4,36)
(4,37)-(4,47)
(4,38)-(4,39)
(4,43)-(4,46)
(4,48)-(4,55)
(4,49)-(4,50)
(4,53)-(4,54)
(5,2)-(5,15)
(5,2)-(5,10)
(5,11)-(5,13)
(5,14)-(5,15)
(7,12)-(13,44)
(7,15)-(13,44)
(8,2)-(13,44)
(8,13)-(8,27)
(8,13)-(8,24)
(8,25)-(8,27)
(9,2)-(13,44)
(9,13)-(9,27)
(9,13)-(9,24)
(9,25)-(9,27)
(10,2)-(13,44)
(10,13)-(10,24)
(10,13)-(10,17)
(10,20)-(10,24)
(11,2)-(13,44)
(11,5)-(11,13)
(11,5)-(11,9)
(11,12)-(11,13)
(12,7)-(12,48)
(12,8)-(12,43)
(12,9)-(12,20)
(12,21)-(12,39)
(12,22)-(12,27)
(12,28)-(12,29)
(12,30)-(12,38)
(12,33)-(12,37)
(12,40)-(12,42)
(12,45)-(12,47)
(13,7)-(13,44)
(13,8)-(13,10)
(13,12)-(13,43)
(13,13)-(13,24)
(13,25)-(13,39)
(13,26)-(13,31)
(13,32)-(13,33)
(13,34)-(13,38)
(13,40)-(13,42)
(15,19)-(16,71)
(16,2)-(16,71)
(16,8)-(16,9)
(16,23)-(16,25)
(16,37)-(16,71)
(16,40)-(16,45)
(16,40)-(16,41)
(16,44)-(16,45)
(16,51)-(16,64)
(16,51)-(16,61)
(16,62)-(16,64)
(16,70)-(16,71)
(18,11)-(28,34)
(18,14)-(28,34)
(19,2)-(28,34)
(19,11)-(27,51)
(20,4)-(27,51)
(20,10)-(24,77)
(20,12)-(24,77)
(21,6)-(24,77)
(21,12)-(21,13)
(22,22)-(22,34)
(22,23)-(22,30)
(22,24)-(22,26)
(22,28)-(22,29)
(22,32)-(22,33)
(24,10)-(24,77)
(24,20)-(24,35)
(24,20)-(24,27)
(24,21)-(24,22)
(24,25)-(24,26)
(24,30)-(24,35)
(24,39)-(24,77)
(24,40)-(24,55)
(24,41)-(24,42)
(24,44)-(24,54)
(24,45)-(24,48)
(24,51)-(24,53)
(24,57)-(24,76)
(24,58)-(24,70)
(24,59)-(24,62)
(24,67)-(24,69)
(24,74)-(24,75)
(25,4)-(27,51)
(25,15)-(25,46)
(25,16)-(25,41)
(25,17)-(25,18)
(25,22)-(25,40)
(25,23)-(25,36)
(25,24)-(25,32)
(25,33)-(25,35)
(25,38)-(25,39)
(25,43)-(25,45)
(26,4)-(27,51)
(26,15)-(26,33)
(26,15)-(26,16)
(26,20)-(26,33)
(26,21)-(26,29)
(26,30)-(26,32)
(27,4)-(27,51)
(27,18)-(27,44)
(27,18)-(27,32)
(27,33)-(27,34)
(27,35)-(27,39)
(27,40)-(27,44)
(27,48)-(27,51)
(28,2)-(28,34)
(28,2)-(28,12)
(28,13)-(28,34)
(28,14)-(28,17)
(28,18)-(28,33)
(28,19)-(28,26)
(28,27)-(28,29)
(28,30)-(28,32)
*)
