
let rec clone x n =
  match n > 0 with | true  -> x :: (clone x (n - 1)) | false  -> [];;

let padZero l1 l2 =
  let length1 = List.length l1 in
  let length2 = List.length l2 in
  match length1 >= length2 with
  | true  ->
      let n = length1 - length2 in
      let zeroes = clone 0 n in (l1, (List.append (zeroes l2)))
  | false  ->
      let n = length2 - length1 in
      let zeroes = clone 0 n in List.append ((zeroes l1), l2);;


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

*)

(* changed spans
(11,37)-(11,62)
(11,50)-(11,61)
(14,32)-(14,43)
(14,32)-(14,61)
(14,45)-(14,56)
(14,46)-(14,52)
*)

(* type error slice
(3,30)-(3,52)
(3,35)-(3,52)
(3,36)-(3,41)
(11,6)-(11,63)
(11,19)-(11,24)
(11,19)-(11,28)
(11,50)-(11,61)
(11,51)-(11,57)
(14,6)-(14,61)
(14,19)-(14,24)
(14,19)-(14,28)
(14,32)-(14,43)
(14,32)-(14,61)
(14,44)-(14,61)
(14,45)-(14,56)
(14,46)-(14,52)
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
(10,6)-(11,63)
(10,14)-(10,31)
(10,14)-(10,21)
(10,24)-(10,31)
(11,6)-(11,63)
(11,19)-(11,28)
(11,19)-(11,24)
(11,25)-(11,26)
(11,27)-(11,28)
(11,32)-(11,63)
(11,33)-(11,35)
(11,37)-(11,62)
(11,38)-(11,49)
(11,50)-(11,61)
(11,51)-(11,57)
(11,58)-(11,60)
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
(14,32)-(14,43)
(14,44)-(14,61)
(14,45)-(14,56)
(14,46)-(14,52)
(14,53)-(14,55)
(14,58)-(14,60)
*)
