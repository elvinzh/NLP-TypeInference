
let rec clone x n =
  if n < 1 then [] else (match n with | _ -> x :: (clone x (n - 1)));;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 = s2
  then (l1, l2)
  else
    if s1 > s2
    then (l1, ((clone 0 (s1 - s2)) @ l2))
    else (((clone 0 (s2 - s1)) @ l1), l2);;

let rec removeZero l =
  if l = []
  then []
  else (let h::t = l in match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      match a1 with
      | [] -> ((a2 @ [(x1 + x2) mod 10]), (a1 @ [(x1 + x2) / 10]))
      | _ -> ((a1 @ 3), (a2 @ 5)) in
    let base = ([], []) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n =
  if n < 1 then [] else (match n with | _ -> x :: (clone x (n - 1)));;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 = s2
  then (l1, l2)
  else
    if s1 > s2
    then (l1, ((clone 0 (s1 - s2)) @ l2))
    else (((clone 0 (s2 - s1)) @ l1), l2);;

let rec removeZero l =
  if l = []
  then []
  else (let h::t = l in match h with | 0 -> removeZero t | _ -> l);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      match a1 with
      | [] -> ((a2 @ [(x1 + x2) mod 10]), (a1 @ [(x1 + x2) / 10]))
      | _ -> ((a1 @ [3]), (a2 @ [5])) in
    let base = ([], []) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(27,20)-(27,21)
(27,30)-(27,31)
*)

(* type error slice
(27,14)-(27,22)
(27,18)-(27,19)
(27,20)-(27,21)
(27,24)-(27,32)
(27,28)-(27,29)
(27,30)-(27,31)
*)

(* all spans
(2,14)-(3,68)
(2,16)-(3,68)
(3,2)-(3,68)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(3,16)-(3,18)
(3,24)-(3,68)
(3,31)-(3,32)
(3,45)-(3,67)
(3,45)-(3,46)
(3,50)-(3,67)
(3,51)-(3,56)
(3,57)-(3,58)
(3,59)-(3,66)
(3,60)-(3,61)
(3,64)-(3,65)
(5,12)-(13,41)
(5,15)-(13,41)
(6,2)-(13,41)
(6,11)-(6,25)
(6,11)-(6,22)
(6,23)-(6,25)
(7,2)-(13,41)
(7,11)-(7,25)
(7,11)-(7,22)
(7,23)-(7,25)
(8,2)-(13,41)
(8,5)-(8,12)
(8,5)-(8,7)
(8,10)-(8,12)
(9,7)-(9,15)
(9,8)-(9,10)
(9,12)-(9,14)
(11,4)-(13,41)
(11,7)-(11,14)
(11,7)-(11,9)
(11,12)-(11,14)
(12,9)-(12,41)
(12,10)-(12,12)
(12,14)-(12,40)
(12,35)-(12,36)
(12,15)-(12,34)
(12,16)-(12,21)
(12,22)-(12,23)
(12,24)-(12,33)
(12,25)-(12,27)
(12,30)-(12,32)
(12,37)-(12,39)
(13,9)-(13,41)
(13,10)-(13,36)
(13,31)-(13,32)
(13,11)-(13,30)
(13,12)-(13,17)
(13,18)-(13,19)
(13,20)-(13,29)
(13,21)-(13,23)
(13,26)-(13,28)
(13,33)-(13,35)
(13,38)-(13,40)
(15,19)-(18,66)
(16,2)-(18,66)
(16,5)-(16,11)
(16,5)-(16,6)
(16,9)-(16,11)
(17,7)-(17,9)
(18,7)-(18,66)
(18,19)-(18,20)
(18,24)-(18,65)
(18,30)-(18,31)
(18,44)-(18,56)
(18,44)-(18,54)
(18,55)-(18,56)
(18,64)-(18,65)
(20,11)-(31,34)
(20,14)-(31,34)
(21,2)-(31,34)
(21,11)-(30,51)
(22,4)-(30,51)
(22,10)-(27,33)
(22,12)-(27,33)
(23,6)-(27,33)
(23,20)-(23,21)
(24,6)-(27,33)
(24,20)-(24,21)
(25,6)-(27,33)
(25,12)-(25,14)
(26,14)-(26,66)
(26,15)-(26,40)
(26,19)-(26,20)
(26,16)-(26,18)
(26,21)-(26,39)
(26,22)-(26,38)
(26,22)-(26,31)
(26,23)-(26,25)
(26,28)-(26,30)
(26,36)-(26,38)
(26,42)-(26,65)
(26,46)-(26,47)
(26,43)-(26,45)
(26,48)-(26,64)
(26,49)-(26,63)
(26,49)-(26,58)
(26,50)-(26,52)
(26,55)-(26,57)
(26,61)-(26,63)
(27,13)-(27,33)
(27,14)-(27,22)
(27,18)-(27,19)
(27,15)-(27,17)
(27,20)-(27,21)
(27,24)-(27,32)
(27,28)-(27,29)
(27,25)-(27,27)
(27,30)-(27,31)
(28,4)-(30,51)
(28,15)-(28,23)
(28,16)-(28,18)
(28,20)-(28,22)
(29,4)-(30,51)
(29,15)-(29,33)
(29,15)-(29,27)
(29,28)-(29,30)
(29,31)-(29,33)
(30,4)-(30,51)
(30,18)-(30,44)
(30,18)-(30,32)
(30,33)-(30,34)
(30,35)-(30,39)
(30,40)-(30,44)
(30,48)-(30,51)
(31,2)-(31,34)
(31,2)-(31,12)
(31,13)-(31,34)
(31,14)-(31,17)
(31,18)-(31,33)
(31,19)-(31,26)
(31,27)-(31,29)
(31,30)-(31,32)
*)
