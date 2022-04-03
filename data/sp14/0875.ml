
let rec clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2
  then (l1, ((clone 0 (len1 - len2)) @ l2))
  else (((clone 0 (len2 - len1)) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | 0::t -> removeZero t | t -> t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      let xx = (x1 + x2) + a1 in
      if xx > 9 then (1, ((xx - 10) :: a2)) else (0, (xx :: a2)) in
    let base = (0, []) in
    let args = List.combine (List.rev (0 :: l1)) (List.rev (0 :: l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  let rec helper i l acc s =
    let l2 = List.rev l in
    match l2 with
    | [] -> s :: acc
    | h::t ->
        let n = (i * h) + s in
        if n > 9
        then helper i t ((n mod 10) :: acc) (n / 10)
        else helper i t (n :: acc) 0 in
  helper i l [] 0;;

let bigMul l1 l2 =
  let f a x =
    let (v,l) = x in
    let (ac,la) = a in
    let mul = mulByDigit v l in
    let shift = mulByDigit mul ac in ((ac * 10), (bigAdd shift la)) in
  let base = (1, []) in
  let args = List.map (fun x  -> (x, (List.rev l2))) l1 in
  let (_,res) = List.fold_left f base args in res;;


(* fix

let rec clone x n =
  let rec helper x n acc =
    if n <= 0 then acc else helper x (n - 1) (x :: acc) in
  helper x n [];;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2
  then (l1, ((clone 0 (len1 - len2)) @ l2))
  else (((clone 0 (len2 - len1)) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | 0::t -> removeZero t | t -> t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      let xx = (x1 + x2) + a1 in
      if xx > 9 then (1, ((xx - 10) :: a2)) else (0, (xx :: a2)) in
    let base = (0, []) in
    let args = List.combine (List.rev (0 :: l1)) (List.rev (0 :: l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  let rec helper i l acc s =
    let l2 = List.rev l in
    match l2 with
    | [] -> s :: acc
    | h::t ->
        let n = (i * h) + s in
        if n > 9
        then helper i t ((n mod 10) :: acc) (n / 10)
        else helper i t (n :: acc) 0 in
  helper i l [] 0;;

let bigMul l1 l2 =
  let f a x =
    let (v,l) = x in
    let (ac,la) = a in
    let mul = mulByDigit v l in
    let shift = mulByDigit ac mul in ((ac * 10), (bigAdd shift la)) in
  let base = (1, []) in
  let args = List.map (fun x  -> (x, (List.rev l2))) l1 in
  let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(46,27)-(46,30)
(46,37)-(46,67)
*)

(* type error slice
(29,3)-(39,19)
(29,19)-(39,17)
(29,21)-(39,17)
(30,2)-(39,17)
(32,4)-(38,36)
(33,12)-(33,20)
(35,8)-(38,36)
(35,16)-(35,23)
(35,17)-(35,18)
(36,8)-(38,36)
(37,13)-(37,19)
(37,13)-(37,52)
(37,20)-(37,21)
(38,13)-(38,19)
(38,13)-(38,36)
(39,2)-(39,8)
(39,2)-(39,17)
(39,9)-(39,10)
(45,4)-(46,67)
(45,14)-(45,24)
(45,14)-(45,28)
(46,16)-(46,26)
(46,16)-(46,33)
(46,27)-(46,30)
*)

(* all spans
(2,14)-(5,15)
(2,16)-(5,15)
(3,2)-(5,15)
(3,17)-(4,55)
(3,19)-(4,55)
(3,21)-(4,55)
(4,4)-(4,55)
(4,7)-(4,13)
(4,7)-(4,8)
(4,12)-(4,13)
(4,19)-(4,22)
(4,28)-(4,55)
(4,28)-(4,34)
(4,35)-(4,36)
(4,37)-(4,44)
(4,38)-(4,39)
(4,42)-(4,43)
(4,45)-(4,55)
(4,46)-(4,47)
(4,51)-(4,54)
(5,2)-(5,15)
(5,2)-(5,8)
(5,9)-(5,10)
(5,11)-(5,12)
(5,13)-(5,15)
(7,12)-(12,43)
(7,15)-(12,43)
(8,2)-(12,43)
(8,13)-(8,27)
(8,13)-(8,24)
(8,25)-(8,27)
(9,2)-(12,43)
(9,13)-(9,27)
(9,13)-(9,24)
(9,25)-(9,27)
(10,2)-(12,43)
(10,5)-(10,16)
(10,5)-(10,9)
(10,12)-(10,16)
(11,7)-(11,43)
(11,8)-(11,10)
(11,12)-(11,42)
(11,37)-(11,38)
(11,13)-(11,36)
(11,14)-(11,19)
(11,20)-(11,21)
(11,22)-(11,35)
(11,23)-(11,27)
(11,30)-(11,34)
(11,39)-(11,41)
(12,7)-(12,43)
(12,8)-(12,38)
(12,33)-(12,34)
(12,9)-(12,32)
(12,10)-(12,15)
(12,16)-(12,17)
(12,18)-(12,31)
(12,19)-(12,23)
(12,26)-(12,30)
(12,35)-(12,37)
(12,40)-(12,42)
(14,19)-(15,57)
(15,2)-(15,57)
(15,8)-(15,9)
(15,23)-(15,25)
(15,36)-(15,48)
(15,36)-(15,46)
(15,47)-(15,48)
(15,56)-(15,57)
(17,11)-(27,34)
(17,14)-(27,34)
(18,2)-(27,34)
(18,11)-(26,51)
(19,4)-(26,51)
(19,10)-(23,64)
(19,12)-(23,64)
(20,6)-(23,64)
(20,20)-(20,21)
(21,6)-(23,64)
(21,20)-(21,21)
(22,6)-(23,64)
(22,15)-(22,29)
(22,15)-(22,24)
(22,16)-(22,18)
(22,21)-(22,23)
(22,27)-(22,29)
(23,6)-(23,64)
(23,9)-(23,15)
(23,9)-(23,11)
(23,14)-(23,15)
(23,21)-(23,43)
(23,22)-(23,23)
(23,25)-(23,42)
(23,26)-(23,35)
(23,27)-(23,29)
(23,32)-(23,34)
(23,39)-(23,41)
(23,49)-(23,64)
(23,50)-(23,51)
(23,53)-(23,63)
(23,54)-(23,56)
(23,60)-(23,62)
(24,4)-(26,51)
(24,15)-(24,22)
(24,16)-(24,17)
(24,19)-(24,21)
(25,4)-(26,51)
(25,15)-(25,69)
(25,15)-(25,27)
(25,28)-(25,48)
(25,29)-(25,37)
(25,38)-(25,47)
(25,39)-(25,40)
(25,44)-(25,46)
(25,49)-(25,69)
(25,50)-(25,58)
(25,59)-(25,68)
(25,60)-(25,61)
(25,65)-(25,67)
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
(29,19)-(39,17)
(29,21)-(39,17)
(30,2)-(39,17)
(30,17)-(38,36)
(30,19)-(38,36)
(30,21)-(38,36)
(30,25)-(38,36)
(31,4)-(38,36)
(31,13)-(31,23)
(31,13)-(31,21)
(31,22)-(31,23)
(32,4)-(38,36)
(32,10)-(32,12)
(33,12)-(33,20)
(33,12)-(33,13)
(33,17)-(33,20)
(35,8)-(38,36)
(35,16)-(35,27)
(35,16)-(35,23)
(35,17)-(35,18)
(35,21)-(35,22)
(35,26)-(35,27)
(36,8)-(38,36)
(36,11)-(36,16)
(36,11)-(36,12)
(36,15)-(36,16)
(37,13)-(37,52)
(37,13)-(37,19)
(37,20)-(37,21)
(37,22)-(37,23)
(37,24)-(37,43)
(37,25)-(37,35)
(37,26)-(37,27)
(37,32)-(37,34)
(37,39)-(37,42)
(37,44)-(37,52)
(37,45)-(37,46)
(37,49)-(37,51)
(38,13)-(38,36)
(38,13)-(38,19)
(38,20)-(38,21)
(38,22)-(38,23)
(38,24)-(38,34)
(38,25)-(38,26)
(38,30)-(38,33)
(38,35)-(38,36)
(39,2)-(39,17)
(39,2)-(39,8)
(39,9)-(39,10)
(39,11)-(39,12)
(39,13)-(39,15)
(39,16)-(39,17)
(41,11)-(49,49)
(41,14)-(49,49)
(42,2)-(49,49)
(42,8)-(46,67)
(42,10)-(46,67)
(43,4)-(46,67)
(43,16)-(43,17)
(44,4)-(46,67)
(44,18)-(44,19)
(45,4)-(46,67)
(45,14)-(45,28)
(45,14)-(45,24)
(45,25)-(45,26)
(45,27)-(45,28)
(46,4)-(46,67)
(46,16)-(46,33)
(46,16)-(46,26)
(46,27)-(46,30)
(46,31)-(46,33)
(46,37)-(46,67)
(46,38)-(46,47)
(46,39)-(46,41)
(46,44)-(46,46)
(46,49)-(46,66)
(46,50)-(46,56)
(46,57)-(46,62)
(46,63)-(46,65)
(47,2)-(49,49)
(47,13)-(47,20)
(47,14)-(47,15)
(47,17)-(47,19)
(48,2)-(49,49)
(48,13)-(48,55)
(48,13)-(48,21)
(48,22)-(48,52)
(48,33)-(48,51)
(48,34)-(48,35)
(48,37)-(48,50)
(48,38)-(48,46)
(48,47)-(48,49)
(48,53)-(48,55)
(49,2)-(49,49)
(49,16)-(49,42)
(49,16)-(49,30)
(49,31)-(49,32)
(49,33)-(49,37)
(49,38)-(49,42)
(49,46)-(49,49)
*)
