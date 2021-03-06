
let rec clone x n =
  let rec helper a x n =
    if n <= 0 then a else (let a' = x :: a in helper a' x (n - 1)) in
  helper [] x n;;

let padZero l1 l2 =
  let length1 = List.length l1 in
  let length2 = List.length l2 in
  if length1 > length2
  then (l1, (List.append (clone 0 (length1 - length2)) l2))
  else
    if length2 > length1
    then ((List.append (clone 0 (length2 - length1)) l1), l2)
    else (l1, l2);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (x1,x2) ->
          (match a with
           | (o,z) ->
               if ((o + x1) + x2) > 9
               then (1, ((((o + x1) + x2) mod 10) :: z))
               else (0, (((o + x1) + x2) :: z))) in
    let base = (0, []) in
    let args =
      let rec pair list1 list2 =
        match (list1, list2) with
        | (h1::t1,h2::t2) -> (h1, h2) :: (pair t1 t2)
        | (_,_) -> [] in
      pair List.rev l1 List.rev l2 in
    let (_,res) = List.fold_left f base args in args in
  add (padZero l1 l2);;


(* fix

let rec clone x n =
  let rec helper a x n =
    if n <= 0 then a else (let a' = x :: a in helper a' x (n - 1)) in
  helper [] x n;;

let padZero l1 l2 =
  let length1 = List.length l1 in
  let length2 = List.length l2 in
  if length1 > length2
  then (l1, (List.append (clone 0 (length1 - length2)) l2))
  else
    if length2 > length1
    then ((List.append (clone 0 (length2 - length1)) l1), l2)
    else (l1, l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (x1,x2) ->
          (match a with
           | (o,z) ->
               if ((o + x1) + x2) > 9
               then (1, ((((o + x1) + x2) mod 10) :: z))
               else (0, (((o + x1) + x2) :: z))) in
    let base = (0, []) in
    let args =
      let rec pair list1 list2 =
        match (list1, list2) with
        | (h1::t1,h2::t2) -> (h1, h2) :: (pair t1 t2)
        | (_,_) -> [] in
      pair (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(17,11)-(35,21)
(33,6)-(33,34)
(33,11)-(33,19)
(33,23)-(33,31)
(34,48)-(34,52)
(35,2)-(35,5)
*)

(* type error slice
(31,29)-(31,53)
(31,41)-(31,53)
(31,42)-(31,46)
(33,6)-(33,10)
(33,6)-(33,34)
*)

(* all spans
(2,14)-(5,15)
(2,16)-(5,15)
(3,2)-(5,15)
(3,17)-(4,66)
(3,19)-(4,66)
(3,21)-(4,66)
(4,4)-(4,66)
(4,7)-(4,13)
(4,7)-(4,8)
(4,12)-(4,13)
(4,19)-(4,20)
(4,26)-(4,66)
(4,36)-(4,42)
(4,36)-(4,37)
(4,41)-(4,42)
(4,46)-(4,65)
(4,46)-(4,52)
(4,53)-(4,55)
(4,56)-(4,57)
(4,58)-(4,65)
(4,59)-(4,60)
(4,63)-(4,64)
(5,2)-(5,15)
(5,2)-(5,8)
(5,9)-(5,11)
(5,12)-(5,13)
(5,14)-(5,15)
(7,12)-(15,17)
(7,15)-(15,17)
(8,2)-(15,17)
(8,16)-(8,30)
(8,16)-(8,27)
(8,28)-(8,30)
(9,2)-(15,17)
(9,16)-(9,30)
(9,16)-(9,27)
(9,28)-(9,30)
(10,2)-(15,17)
(10,5)-(10,22)
(10,5)-(10,12)
(10,15)-(10,22)
(11,7)-(11,59)
(11,8)-(11,10)
(11,12)-(11,58)
(11,13)-(11,24)
(11,25)-(11,54)
(11,26)-(11,31)
(11,32)-(11,33)
(11,34)-(11,53)
(11,35)-(11,42)
(11,45)-(11,52)
(11,55)-(11,57)
(13,4)-(15,17)
(13,7)-(13,24)
(13,7)-(13,14)
(13,17)-(13,24)
(14,9)-(14,61)
(14,10)-(14,56)
(14,11)-(14,22)
(14,23)-(14,52)
(14,24)-(14,29)
(14,30)-(14,31)
(14,32)-(14,51)
(14,33)-(14,40)
(14,43)-(14,50)
(14,53)-(14,55)
(14,58)-(14,60)
(15,9)-(15,17)
(15,10)-(15,12)
(15,14)-(15,16)
(17,11)-(35,21)
(17,14)-(35,21)
(18,2)-(35,21)
(18,11)-(34,52)
(19,4)-(34,52)
(19,10)-(26,48)
(19,12)-(26,48)
(20,6)-(26,48)
(20,12)-(20,13)
(22,10)-(26,48)
(22,17)-(22,18)
(24,15)-(26,47)
(24,18)-(24,37)
(24,18)-(24,33)
(24,19)-(24,27)
(24,20)-(24,21)
(24,24)-(24,26)
(24,30)-(24,32)
(24,36)-(24,37)
(25,20)-(25,56)
(25,21)-(25,22)
(25,24)-(25,55)
(25,25)-(25,49)
(25,26)-(25,41)
(25,27)-(25,35)
(25,28)-(25,29)
(25,32)-(25,34)
(25,38)-(25,40)
(25,46)-(25,48)
(25,53)-(25,54)
(26,20)-(26,47)
(26,21)-(26,22)
(26,24)-(26,46)
(26,25)-(26,40)
(26,26)-(26,34)
(26,27)-(26,28)
(26,31)-(26,33)
(26,37)-(26,39)
(26,44)-(26,45)
(27,4)-(34,52)
(27,15)-(27,22)
(27,16)-(27,17)
(27,19)-(27,21)
(28,4)-(34,52)
(29,6)-(33,34)
(29,19)-(32,21)
(29,25)-(32,21)
(30,8)-(32,21)
(30,14)-(30,28)
(30,15)-(30,20)
(30,22)-(30,27)
(31,29)-(31,53)
(31,29)-(31,37)
(31,30)-(31,32)
(31,34)-(31,36)
(31,41)-(31,53)
(31,42)-(31,46)
(31,47)-(31,49)
(31,50)-(31,52)
(32,19)-(32,21)
(33,6)-(33,34)
(33,6)-(33,10)
(33,11)-(33,19)
(33,20)-(33,22)
(33,23)-(33,31)
(33,32)-(33,34)
(34,4)-(34,52)
(34,18)-(34,44)
(34,18)-(34,32)
(34,33)-(34,34)
(34,35)-(34,39)
(34,40)-(34,44)
(34,48)-(34,52)
(35,2)-(35,21)
(35,2)-(35,5)
(35,6)-(35,21)
(35,7)-(35,14)
(35,15)-(35,17)
(35,18)-(35,20)
*)
