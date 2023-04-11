function solution(players, callings) {
  const nameObj = {}
  const numObj = {}
  players.forEach((v, i) => {
    nameObj[v] = i
    numObj[i] = v
  })

  callings.forEach((newPerson) => {
    const backNum = nameObj[newPerson]
    const frontNum = backNum - 1
    const old = numObj[frontNum]

    nameObj[newPerson] = frontNum
    numObj[frontNum] = newPerson
    nameObj[old] = backNum
    numObj[backNum] = old
  })

  // console.log(nameObj)
}
