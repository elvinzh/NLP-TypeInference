
let digitsOfInt n =
  if n < 0
  then []
  else
    (let rec loop n acc =
       if n = 0 then acc else loop (n / 10) ((n mod 10) :: acc) in
     match n with | 0 -> [0] | _ -> loop n []);;

let digits n = digitsOfInt (abs n);;

let rec sumList xs = match xs with | [] -> 0 | h::t -> h + (sumList t);;

let rec digitalRoot n =
  let x = sumList (digits n) in if x > 9 then digitalRoot x else sumList x;;


(* fix

let digitsOfInt n =
  if n < 0
  then []
  else
    (let rec loop n acc =
       if n = 0 then acc else loop (n / 10) ((n mod 10) :: acc) in
     match n with | 0 -> [0] | _ -> loop n []);;

let digits n = digitsOfInt (abs n);;

let rec sumList xs = match xs with | [] -> 0 | h::t -> h + (sumList t);;

let rec digitalRoot n =
  if (sumList (digits n)) > 9
  then digitalRoot (sumList (digits n))
  else sumList (digits n);;

*)

(* changed spans
(15,2)-(15,74)
(15,10)-(15,28)
(15,32)-(15,74)
(15,35)-(15,36)
(15,35)-(15,40)
(15,58)-(15,59)
(15,73)-(15,74)
*)

(* type error slice
(12,21)-(12,70)
(12,55)-(12,70)
(12,59)-(12,70)
(12,60)-(12,67)
(12,68)-(12,69)
(15,2)-(15,74)
(15,10)-(15,17)
(15,10)-(15,28)
(15,65)-(15,72)
(15,65)-(15,74)
(15,73)-(15,74)
*)

(* all spans
(2,16)-(8,46)
(3,2)-(8,46)
(3,5)-(3,10)
(3,5)-(3,6)
(3,9)-(3,10)
(4,7)-(4,9)
(6,4)-(8,46)
(6,18)-(7,63)
(6,20)-(7,63)
(7,7)-(7,63)
(7,10)-(7,15)
(7,10)-(7,11)
(7,14)-(7,15)
(7,21)-(7,24)
(7,30)-(7,63)
(7,30)-(7,34)
(7,35)-(7,43)
(7,36)-(7,37)
(7,40)-(7,42)
(7,44)-(7,63)
(7,45)-(7,55)
(7,46)-(7,47)
(7,52)-(7,54)
(7,59)-(7,62)
(8,5)-(8,45)
(8,11)-(8,12)
(8,25)-(8,28)
(8,26)-(8,27)
(8,36)-(8,45)
(8,36)-(8,40)
(8,41)-(8,42)
(8,43)-(8,45)
(10,11)-(10,34)
(10,15)-(10,34)
(10,15)-(10,26)
(10,27)-(10,34)
(10,28)-(10,31)
(10,32)-(10,33)
(12,16)-(12,70)
(12,21)-(12,70)
(12,27)-(12,29)
(12,43)-(12,44)
(12,55)-(12,70)
(12,55)-(12,56)
(12,59)-(12,70)
(12,60)-(12,67)
(12,68)-(12,69)
(14,20)-(15,74)
(15,2)-(15,74)
(15,10)-(15,28)
(15,10)-(15,17)
(15,18)-(15,28)
(15,19)-(15,25)
(15,26)-(15,27)
(15,32)-(15,74)
(15,35)-(15,40)
(15,35)-(15,36)
(15,39)-(15,40)
(15,46)-(15,59)
(15,46)-(15,57)
(15,58)-(15,59)
(15,65)-(15,74)
(15,65)-(15,72)
(15,73)-(15,74)
*)
