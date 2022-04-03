
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
  | h::t -> (match h with | 0 -> removeZero t | _ -> t);;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (_,h2::t2) ->
          (match x with | [] -> (t2, [h2]) | h3::t3 -> (t2, (h2 :: h3 :: t3))) in
    let base = (l1, []) in
    let args = l2 in let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


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
  | h::t -> (match h with | 0 -> removeZero t | _ -> t);;

let bigAdd l1 l2 =
  let add (l1,l2) = [0; 0; 0; 0; 0] in removeZero (add (padZero l1 l2));;

*)

(* changed spans
(23,4)-(28,68)
(23,10)-(26,78)
(23,12)-(26,78)
(24,6)-(26,78)
(24,12)-(24,13)
(26,10)-(26,78)
(26,17)-(26,18)
(26,32)-(26,42)
(26,33)-(26,35)
(26,37)-(26,41)
(26,38)-(26,40)
(26,55)-(26,77)
(26,56)-(26,58)
(26,60)-(26,76)
(26,61)-(26,63)
(26,67)-(26,69)
(26,67)-(26,75)
(26,73)-(26,75)
(27,4)-(28,68)
(27,15)-(27,23)
(27,16)-(27,18)
(27,20)-(27,22)
(28,4)-(28,68)
(28,15)-(28,17)
(28,21)-(28,68)
(28,35)-(28,49)
(28,35)-(28,61)
(28,50)-(28,51)
(28,52)-(28,56)
(28,57)-(28,61)
(28,65)-(28,68)
(29,2)-(29,34)
*)

(* type error slice
(5,3)-(14,63)
(5,12)-(14,61)
(5,15)-(14,61)
(6,2)-(14,61)
(7,2)-(14,61)
(8,2)-(14,61)
(10,6)-(11,61)
(11,6)-(11,61)
(11,19)-(11,24)
(11,19)-(11,28)
(11,32)-(11,61)
(11,33)-(11,35)
(11,37)-(11,60)
(11,38)-(11,49)
(11,50)-(11,56)
(14,6)-(14,61)
(14,19)-(14,24)
(14,19)-(14,28)
(14,33)-(14,56)
(14,34)-(14,45)
(14,46)-(14,52)
(14,53)-(14,55)
(22,2)-(29,34)
(22,11)-(28,68)
(23,4)-(28,68)
(23,10)-(26,78)
(23,12)-(26,78)
(24,6)-(26,78)
(26,10)-(26,78)
(26,17)-(26,18)
(26,32)-(26,42)
(26,33)-(26,35)
(26,60)-(26,76)
(26,61)-(26,63)
(26,67)-(26,69)
(26,67)-(26,75)
(27,4)-(28,68)
(27,15)-(27,23)
(27,16)-(27,18)
(28,4)-(28,68)
(28,15)-(28,17)
(28,35)-(28,49)
(28,35)-(28,61)
(28,50)-(28,51)
(28,52)-(28,56)
(28,57)-(28,61)
(29,13)-(29,34)
(29,14)-(29,17)
(29,18)-(29,33)
(29,19)-(29,26)
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
(16,19)-(19,55)
(17,2)-(19,55)
(17,8)-(17,9)
(18,10)-(18,12)
(19,12)-(19,55)
(19,19)-(19,20)
(19,33)-(19,45)
(19,33)-(19,43)
(19,44)-(19,45)
(19,53)-(19,54)
(21,11)-(29,34)
(21,14)-(29,34)
(22,2)-(29,34)
(22,11)-(28,68)
(23,4)-(28,68)
(23,10)-(26,78)
(23,12)-(26,78)
(24,6)-(26,78)
(24,12)-(24,13)
(26,10)-(26,78)
(26,17)-(26,18)
(26,32)-(26,42)
(26,33)-(26,35)
(26,37)-(26,41)
(26,38)-(26,40)
(26,55)-(26,77)
(26,56)-(26,58)
(26,60)-(26,76)
(26,61)-(26,63)
(26,67)-(26,75)
(26,67)-(26,69)
(26,73)-(26,75)
(27,4)-(28,68)
(27,15)-(27,23)
(27,16)-(27,18)
(27,20)-(27,22)
(28,4)-(28,68)
(28,15)-(28,17)
(28,21)-(28,68)
(28,35)-(28,61)
(28,35)-(28,49)
(28,50)-(28,51)
(28,52)-(28,56)
(28,57)-(28,61)
(28,65)-(28,68)
(29,2)-(29,34)
(29,2)-(29,12)
(29,13)-(29,34)
(29,14)-(29,17)
(29,18)-(29,33)
(29,19)-(29,26)
(29,27)-(29,29)
(29,30)-(29,32)
*)