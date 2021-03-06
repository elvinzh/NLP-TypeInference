
let rec clone x n =
  match n > 0 with | true  -> x :: (clone x (n - 1)) | false  -> [];;

let padZero l1 l2 =
  let length1 = List.length l1 in
  let length2 = List.length l2 in
  match length1 >= length2 with
  | true  ->
      let n = length1 - length2 in
      let zeroes = clone 0 n in (l1, (List.append zeroes l2))
  | false  ->
      let n = length2 - length1 in
      let zeroes = clone 0 n in ((List.append zeroes l1), l2);;

let rec removeZero l =
  match l with
  | [] -> []
  | h::t -> (match h with | 0 -> removeZero t | _ -> h :: t);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (h1::t1,rh::rt) ->
          (t1, ((((h1 + x) + rh) / 10) :: (((h1 + x) + rh) mod 10) :: rt))
      | _ -> ([], []) in
    let base = ((List.rev l1), [0]) in
    let args = List.rev l2 in let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  match i mod 2 with
  | 0 ->
      (match i with
       | 0 -> []
       | 2 -> bigAdd l l
       | _ -> bigAdd (mulByDigit (i / 2) l) (mulByDigit (i / 2) l))
  | _ -> (match i with | 1 -> l | _ -> bigAdd l (mulByDigit (i - 1) l));;

let bigMul l1 l2 =
  let f a x =
    match a with
    | (h1::t1,rh::rt) -> (t1, ((bigAdd (mulByDigit x rh) rh) :: rt)) in
  let base = ((List.rev l1), [1]) in
  let args = List.rev l2 in let (_,res) = List.fold_left f base args in res;;


(* fix

let rec clone x n =
  match n > 0 with | true  -> x :: (clone x (n - 1)) | false  -> [];;

let padZero l1 l2 =
  let length1 = List.length l1 in
  let length2 = List.length l2 in
  match length1 >= length2 with
  | true  ->
      let n = length1 - length2 in
      let zeroes = clone 0 n in (l1, (List.append zeroes l2))
  | false  ->
      let n = length2 - length1 in
      let zeroes = clone 0 n in ((List.append zeroes l1), l2);;

let rec removeZero l =
  match l with
  | [] -> []
  | h::t -> (match h with | 0 -> removeZero t | _ -> h :: t);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (h1::t1,rh::rt) ->
          (t1, ((((h1 + x) + rh) / 10) :: (((h1 + x) + rh) mod 10) :: rt))
      | _ -> ([], []) in
    let base = ((List.rev l1), [0]) in
    let args = List.rev l2 in let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  match i mod 2 with
  | 0 ->
      (match i with
       | 0 -> []
       | 2 -> bigAdd l l
       | _ -> bigAdd (mulByDigit (i / 2) l) (mulByDigit (i / 2) l))
  | _ -> (match i with | 1 -> l | _ -> bigAdd l (mulByDigit (i - 1) l));;

let bigMul l1 l2 =
  let f a x =
    match a with
    | (h1::t1,r) -> (t1, (bigAdd (mulByDigit h1 (List.append r [0])) r)) in
  let base = (l1, []) in
  let args = l2 in let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(43,4)-(44,68)
(44,30)-(44,67)
(44,51)-(44,52)
(44,53)-(44,55)
(44,57)-(44,59)
(44,64)-(44,66)
(45,2)-(46,75)
(45,14)-(45,27)
(45,15)-(45,23)
(45,29)-(45,32)
(45,30)-(45,31)
(46,13)-(46,21)
(46,13)-(46,24)
*)

(* type error slice
(19,12)-(19,60)
(19,33)-(19,43)
(19,33)-(19,45)
(19,53)-(19,59)
(21,3)-(30,36)
(21,11)-(30,34)
(21,14)-(30,34)
(22,2)-(30,34)
(30,2)-(30,12)
(30,2)-(30,34)
(42,2)-(46,75)
(42,8)-(44,68)
(42,10)-(44,68)
(43,4)-(44,68)
(44,25)-(44,68)
(44,30)-(44,67)
(44,31)-(44,60)
(44,32)-(44,38)
(45,2)-(46,75)
(45,13)-(45,33)
(45,29)-(45,32)
(45,30)-(45,31)
(46,42)-(46,56)
(46,42)-(46,68)
(46,57)-(46,58)
(46,59)-(46,63)
*)

(* all spans
(2,14)-(3,67)
(2,16)-(3,67)
(3,2)-(3,67)
(3,8)-(3,13)
(3,8)-(3,9)
(3,12)-(3,13)
(3,30)-(3,52)
(3,30)-(3,31)
(3,35)-(3,52)
(3,36)-(3,41)
(3,42)-(3,43)
(3,44)-(3,51)
(3,45)-(3,46)
(3,49)-(3,50)
(3,65)-(3,67)
(5,12)-(14,61)
(5,15)-(14,61)
(6,2)-(14,61)
(6,16)-(6,30)
(6,16)-(6,27)
(6,28)-(6,30)
(7,2)-(14,61)
(7,16)-(7,30)
(7,16)-(7,27)
(7,28)-(7,30)
(8,2)-(14,61)
(8,8)-(8,26)
(8,8)-(8,15)
(8,19)-(8,26)
(10,6)-(11,61)
(10,14)-(10,31)
(10,14)-(10,21)
(10,24)-(10,31)
(11,6)-(11,61)
(11,19)-(11,28)
(11,19)-(11,24)
(11,25)-(11,26)
(11,27)-(11,28)
(11,32)-(11,61)
(11,33)-(11,35)
(11,37)-(11,60)
(11,38)-(11,49)
(11,50)-(11,56)
(11,57)-(11,59)
(13,6)-(14,61)
(13,14)-(13,31)
(13,14)-(13,21)
(13,24)-(13,31)
(14,6)-(14,61)
(14,19)-(14,28)
(14,19)-(14,24)
(14,25)-(14,26)
(14,27)-(14,28)
(14,32)-(14,61)
(14,33)-(14,56)
(14,34)-(14,45)
(14,46)-(14,52)
(14,53)-(14,55)
(14,58)-(14,60)
(16,19)-(19,60)
(17,2)-(19,60)
(17,8)-(17,9)
(18,10)-(18,12)
(19,12)-(19,60)
(19,19)-(19,20)
(19,33)-(19,45)
(19,33)-(19,43)
(19,44)-(19,45)
(19,53)-(19,59)
(19,53)-(19,54)
(19,58)-(19,59)
(21,11)-(30,34)
(21,14)-(30,34)
(22,2)-(30,34)
(22,11)-(29,77)
(23,4)-(29,77)
(23,10)-(27,21)
(23,12)-(27,21)
(24,6)-(27,21)
(24,12)-(24,13)
(26,10)-(26,74)
(26,11)-(26,13)
(26,15)-(26,73)
(26,16)-(26,38)
(26,17)-(26,32)
(26,18)-(26,26)
(26,19)-(26,21)
(26,24)-(26,25)
(26,29)-(26,31)
(26,35)-(26,37)
(26,42)-(26,72)
(26,42)-(26,66)
(26,43)-(26,58)
(26,44)-(26,52)
(26,45)-(26,47)
(26,50)-(26,51)
(26,55)-(26,57)
(26,63)-(26,65)
(26,70)-(26,72)
(27,13)-(27,21)
(27,14)-(27,16)
(27,18)-(27,20)
(28,4)-(29,77)
(28,15)-(28,35)
(28,16)-(28,29)
(28,17)-(28,25)
(28,26)-(28,28)
(28,31)-(28,34)
(28,32)-(28,33)
(29,4)-(29,77)
(29,15)-(29,26)
(29,15)-(29,23)
(29,24)-(29,26)
(29,30)-(29,77)
(29,44)-(29,70)
(29,44)-(29,58)
(29,59)-(29,60)
(29,61)-(29,65)
(29,66)-(29,70)
(29,74)-(29,77)
(30,2)-(30,34)
(30,2)-(30,12)
(30,13)-(30,34)
(30,14)-(30,17)
(30,18)-(30,33)
(30,19)-(30,26)
(30,27)-(30,29)
(30,30)-(30,32)
(32,19)-(39,71)
(32,21)-(39,71)
(33,2)-(39,71)
(33,8)-(33,15)
(33,8)-(33,9)
(33,14)-(33,15)
(35,6)-(38,67)
(35,13)-(35,14)
(36,14)-(36,16)
(37,14)-(37,24)
(37,14)-(37,20)
(37,21)-(37,22)
(37,23)-(37,24)
(38,14)-(38,66)
(38,14)-(38,20)
(38,21)-(38,43)
(38,22)-(38,32)
(38,33)-(38,40)
(38,34)-(38,35)
(38,38)-(38,39)
(38,41)-(38,42)
(38,44)-(38,66)
(38,45)-(38,55)
(38,56)-(38,63)
(38,57)-(38,58)
(38,61)-(38,62)
(38,64)-(38,65)
(39,9)-(39,71)
(39,16)-(39,17)
(39,30)-(39,31)
(39,39)-(39,70)
(39,39)-(39,45)
(39,46)-(39,47)
(39,48)-(39,70)
(39,49)-(39,59)
(39,60)-(39,67)
(39,61)-(39,62)
(39,65)-(39,66)
(39,68)-(39,69)
(41,11)-(46,75)
(41,14)-(46,75)
(42,2)-(46,75)
(42,8)-(44,68)
(42,10)-(44,68)
(43,4)-(44,68)
(43,10)-(43,11)
(44,25)-(44,68)
(44,26)-(44,28)
(44,30)-(44,67)
(44,31)-(44,60)
(44,32)-(44,38)
(44,39)-(44,56)
(44,40)-(44,50)
(44,51)-(44,52)
(44,53)-(44,55)
(44,57)-(44,59)
(44,64)-(44,66)
(45,2)-(46,75)
(45,13)-(45,33)
(45,14)-(45,27)
(45,15)-(45,23)
(45,24)-(45,26)
(45,29)-(45,32)
(45,30)-(45,31)
(46,2)-(46,75)
(46,13)-(46,24)
(46,13)-(46,21)
(46,22)-(46,24)
(46,28)-(46,75)
(46,42)-(46,68)
(46,42)-(46,56)
(46,57)-(46,58)
(46,59)-(46,63)
(46,64)-(46,68)
(46,72)-(46,75)
*)
