
let rec digitsOfInt n =
  let return = [n mod 10] in
  if (n / 10) <> 0
  then ((n mod 10) :: return; (digitsOfInt (n / 10)) @ return)
  else return;;

let rec sumList xs = match xs with | [] -> 0 | h::t -> h + (sumList t);;

let rec digitalRoot n =
  let digits = digitsOfInt n in
  let s = sumList digits in if (n / 10) <> 0 then digitalRoot else digits;;


(* fix

let rec digitsOfInt n =
  let return = [n mod 10] in
  if (n / 10) <> 0
  then ((n mod 10) :: return; (digitsOfInt (n / 10)) @ return)
  else return;;

let digits n = digitsOfInt (abs n);;

let rec sumList xs = match xs with | [] -> 0 | h::t -> h + (sumList t);;

let rec digitalRoot n =
  let d = digits n in
  let s = sumList d in if (n / 10) <> 0 then digitalRoot s else s;;

*)

(* changed spans
(8,16)-(8,70)
(11,2)-(12,73)
(11,15)-(11,26)
(12,18)-(12,24)
(12,50)-(12,61)
(12,67)-(12,73)
*)

(* type error slice
(5,7)-(5,62)
(5,8)-(5,28)
(10,3)-(12,75)
(10,20)-(12,73)
(11,2)-(12,73)
(12,2)-(12,73)
(12,28)-(12,73)
(12,50)-(12,61)
*)

(* all spans
(2,20)-(6,13)
(3,2)-(6,13)
(3,15)-(3,25)
(3,16)-(3,24)
(3,16)-(3,17)
(3,22)-(3,24)
(4,2)-(6,13)
(4,5)-(4,18)
(4,5)-(4,13)
(4,6)-(4,7)
(4,10)-(4,12)
(4,17)-(4,18)
(5,7)-(5,62)
(5,8)-(5,28)
(5,8)-(5,18)
(5,9)-(5,10)
(5,15)-(5,17)
(5,22)-(5,28)
(5,30)-(5,61)
(5,53)-(5,54)
(5,30)-(5,52)
(5,31)-(5,42)
(5,43)-(5,51)
(5,44)-(5,45)
(5,48)-(5,50)
(5,55)-(5,61)
(6,7)-(6,13)
(8,16)-(8,70)
(8,21)-(8,70)
(8,27)-(8,29)
(8,43)-(8,44)
(8,55)-(8,70)
(8,55)-(8,56)
(8,59)-(8,70)
(8,60)-(8,67)
(8,68)-(8,69)
(10,20)-(12,73)
(11,2)-(12,73)
(11,15)-(11,28)
(11,15)-(11,26)
(11,27)-(11,28)
(12,2)-(12,73)
(12,10)-(12,24)
(12,10)-(12,17)
(12,18)-(12,24)
(12,28)-(12,73)
(12,31)-(12,44)
(12,31)-(12,39)
(12,32)-(12,33)
(12,36)-(12,38)
(12,43)-(12,44)
(12,50)-(12,61)
(12,67)-(12,73)
*)